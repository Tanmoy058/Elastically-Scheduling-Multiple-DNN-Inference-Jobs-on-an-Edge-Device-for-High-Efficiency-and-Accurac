#include <bits/stdc++.h>

using namespace std;

vector<string> parse_by_comma(string shape){
	stringstream ss(shape);
	vector<string> result;

	while( ss.good() )
	{
    	string substr;
    	getline( ss, substr, ',' );
    	result.push_back(substr);
	}
	return result;
}

map<string,int>layer_to_num;
map<int,string>num_to_layer;

vector< vector<int> >dir_graph; //take input can handle this
vector< vector<int> >input_info; //just the reverse graph
vector< vector<int> >just_output_shapes; //for helping with something

void init(int tot_layer){
	vector<int>line;

	for(int i = 0; i<tot_layer+3; ++i){
		dir_graph.push_back(line);
		input_info.push_back(line);
		just_output_shapes.push_back(line);
	}
}

struct layer_info{
	string name;
	long long params;
	int idx;
	vector<long long>output_shape;
	int type; //conv = 1, dense = 2, drop = 3, flat = 4, pool = 5, concat = 6
	double alpha;
	double beta;
	double real;
	long long num_operation;
	vector<long long>input_shape; //all input layers shape ar concated
	//long long filter_size; //only for type 1
	//long long num_filters; //only for type 1
	layer_info(string _name, long long _params, int _idx, vector<long long>_out_shape){
		name = _name;
		params = _params;
		idx = _idx;
		output_shape = _out_shape;
	}
	void calc_things(){
	    cout<<"name: "<<name<<" "<<"type: "<<type<<endl;
		if(type>2){
			alpha = 8;
			long long sm = 0;
			for(int i = 0; i<input_shape.size(); i+=3){
				long long cur = input_shape[i];
				if((i+1)<input_shape.size())cur = cur*input_shape[i+1];
				if((i+2)<input_shape.size())cur = cur*input_shape[i+2];
				sm = sm+cur;
			}
			num_operation = sm;
			real  = (sm*(8.0))/(1024.0*1024.0);
			real = real+3.0*real; //for backward
			beta = max(real*2,alpha*2);
		}
		else if(type==2){
			long long sm = 0;
			for(int i = 0; i<input_shape.size(); i+=3){
				long long cur = input_shape[i];
				if((i+1)<input_shape.size())cur = cur*input_shape[i+1];
				if((i+2)<input_shape.size())cur = cur*input_shape[i+2];
				sm = sm+cur;
			}
			long long sm2 = 0;
			for(int i = 0; i<output_shape.size(); i+=3){
				long long cur = output_shape[i];
				if((i+1)<output_shape.size())cur = cur*output_shape[i+1];
				if((i+2)<output_shape.size())cur = cur*output_shape[i+2];
				sm2 = sm2+cur;
			}
			real = ((sm*sm2)*8.0)/(1024.0*1024.0);
			real = real + 3.0*real;
			alpha = max(8.0,real/2.0);
			beta = max(real*2.0,alpha*2.0);
			num_operation = sm*sm2;
		}
		else{
			//conv layer
			//num_filters is the 3rd dimension of the output channel
			//num_filters = output_shape[2];
			//kernel height and width is the same ... need both input shape and output shape for calculating that
			cout<<"here\n";
			cout<<(int)input_shape.size()<<"\n";
			int input_height = (((int)input_shape.size()==1)?1:input_shape[0]);
			int input_width = (((int)input_shape.size()==2)?1:input_shape[1]);
            cout<<input_height<<" "<<input_width<<"\n";
			int prev_depth_channel = (((int)input_shape.size()==3)?1:input_shape[2]);
			cout<<input_height<<" "<<input_width<<" "<<prev_depth_channel<<endl;
			int next_depth_channel = output_shape[2];

			cout<<input_height<<" "<<input_width<<" "<<prev_depth_channel<<" "<<next_depth_channel<<endl;

			string c_3 = "_c_3";
			string c_5 = "_c_5";

			int kernel_height = 1;
			if(name.find(c_3)!=string::npos)kernel_height = 3;
			if(name.find(c_5)!=string::npos)kernel_height = 5;
			int kernel_width = kernel_height;

			num_operation = input_height*input_width*prev_depth_channel*next_depth_channel*kernel_height*kernel_width;

			//double mem_in = input_width*input_height*prev_depth_channel;
			//double mem_out = output_shape[0]*output_shape[1]*output_shape[2];
			//double mem_inter = kernel_height*kernel_width*prev_depth_channel*output_shape[0]*output_shape[1];

			real = ((num_operation+output_shape[0]*output_shape[1]*output_shape[2])*8.0)/(1024.0*1024.0);
			real = real+3.0*real;
			alpha = max(8.0,real/2.0);
			beta = max(real*2.0,alpha*2.0);
		}
	}
	void calc_type(){
		//type = 1, "conv" or "_c_"
		//type = 2, "dense"
		//type = 3, "drop"
		//type = 4, "flat"
		//type = 5, "pool"
		//type = 6, else
		if(name.find("conv")!=string::npos || name.find("_c_")!=string::npos)type = 1;
		else if(name.find("dense")!=string::npos)type = 2;
		else if(name.find("drop")!=string::npos)type = 3;
		else if(name.find("flat")!=string::npos)type = 4;
		else if(name.find("pool")!=string::npos)type = 5;
		else type = 6;
	}
	void calc_input_shape(){
		for(int v: input_info[idx]){
            for(int o_s: just_output_shapes[v]){
                input_shape.push_back(o_s);
            }
		}
	}
	void print(){
        cout<<name<<" "<<idx<<" "<<type<<" "<<alpha<<" "<<beta<<" "<<real<<" "<<num_operation<<" ";
        for(int ii: input_shape){
            cout<<ii<<",";
        }
        cout<<endl;
	}
};

vector< layer_info > all_layers; //can handle this later

void take_input(int tot_layer){
	ifstream fin("summary_vgg.txt");
	//input_1             (None,28,28,3)
	//int tot_layer = 94;
	int map_idx = 1;

	for(int i = 1; i<=tot_layer; ++i){

		string layer_name;
		string output_shape;
		long long params;
		string input_layers;

		fin>>layer_name>>output_shape>>params;

		if(i>1)fin>>input_layers;

		layer_to_num[layer_name] = map_idx;
		num_to_layer[map_idx] = layer_name;

		string remv = "None,";
		output_shape.erase(output_shape.find(remv),remv.length());
		output_shape.erase(remove(output_shape.begin(),output_shape.end(),'('),output_shape.end());
		output_shape.erase(remove(output_shape.begin(),output_shape.end(),')'),output_shape.end());
        vector<string> output_dims = parse_by_comma(output_shape);
        vector<long long>out_sh;
        for(string s: output_dims)out_sh.push_back((long long)(stoi(s)));
        for(long long ll: out_sh){
            just_output_shapes[map_idx].push_back(ll);
        }

		if(i>1){
            vector<string> all_input_layer = parse_by_comma(input_layers);
            for(string s: all_input_layer){
                dir_graph[layer_to_num[s]].push_back(map_idx);
                input_info[map_idx].push_back(layer_to_num[s]);
            }
		}

        if(i>1){
            layer_info l(layer_name,params,map_idx,out_sh);
            all_layers.push_back(l);
        }
        map_idx++;
	}
	fin.close();
}

void process_all_info(int tot_layer){
    for(int i = 0; i<tot_layer; ++i){
        cout<<i<<"\n";
        all_layers[i].calc_type();
        cout<<"type ber korsi\n";
        all_layers[i].calc_input_shape();
        cout<<"inp shape ber korsi\n";
        all_layers[i].calc_things();
        cout<<"thing ber korsi\n";
    }
}

int main(){
    cout<<"hi\n";
	init(23);
	take_input(23);
	cout<<"inp nisi\n";
    cout<<all_layers.size()<<"\n";

	process_all_info(22);

    double tot_memory = 0.0;

    for(layer_info ll: all_layers){
        ll.print();
        tot_memory = tot_memory+ll.real;
    }
    cout<<tot_memory<<"\n";

    int tot_edges = 0;

    for(int i = 0; i<dir_graph.size(); ++i){
        tot_edges = tot_edges+(int)dir_graph[i].size();
    }

    ofstream fout("ml_graph_vgg.txt");
    fout<<(int)all_layers.size()<<" "<<tot_edges<<"\n";

    for(int i = 0; i<all_layers.size(); ++i){
        fout<<all_layers[i].name<<" "<<all_layers[i].num_operation<<" "<<all_layers[i].alpha<<" "<<all_layers[i].beta<<"\n";
    }

    for(int i = 0; i<dir_graph.size(); ++i){
        string cur_layer = num_to_layer[i];
        for(int j = 0; j<dir_graph[i].size(); ++j){
            string next_layer = num_to_layer[dir_graph[i][j]];
            long long outss = 1;
            for(int k = 0; k<just_output_shapes[i].size(); ++k){
                outss = outss*just_output_shapes[i][k];
            }
            fout<<cur_layer<<" "<<next_layer<<" "<<outss<<"\n";
        }
    }
    fout.close();
	return 0;
}
