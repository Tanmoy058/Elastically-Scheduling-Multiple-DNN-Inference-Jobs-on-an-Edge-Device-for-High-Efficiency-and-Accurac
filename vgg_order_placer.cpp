#include <bits/stdc++.h>
#define MXDEVICE 30
#define MXLAYER 200
#define INF 300000000 //almost 1 day
#define MXMEMORY 8192 //MB
#define WORKER_MEMORY 165
using namespace std;

//ifstream fin("ml_graph.txt");
//ifstream fin("dev_graph.txt");

//ofstream fout("tbgap_map_out.txt");

double alpha;

map<string,int>vertex_map;
map<int,string>rev_vertex_map;
map<string,int>device_map;
map<int,string>rev_device_map;

vector< vector<pair<int,double> > >adj;
vector< vector<pair<int,double> > >rev_adj;
double BW_matrix_master[MXDEVICE]; //in MBps
double t0[MXDEVICE];

struct graph_node{
    string layer_name;
    double num_computation;
    double alpha_memory; //in MB
    double beta_memory; //in MB

    graph_node(string a, double b, double c, double d){
        layer_name = a;
        num_computation = b;
        alpha_memory = c;
        beta_memory = d;
    }
};

struct device_node{
    string device_ip;
    double op_per_time;
    double memory; //in MB

    device_node(string a, double b, double c){
        device_ip = a;
        op_per_time = b;
        memory = c;
    }
};

vector<graph_node>gg;
vector<device_node>dd;
vector<device_node>dd_copy;

void take_input_ml_graph(){
    ifstream fin("ml_graph_vgg.txt");
    ifstream fmemory("vgg_new_memory_calc.txt");
    //format:
    //layers,num_of_connection
    //layer_name, num_com, alpha,beta
    //.....
    //layer_name_1 layer_name_2 amount of data transfer...

    int n,m;
    fin>>n>>m;

    double tot_memory = 0.0;
    for(int i = 1; i<n+3; ++i){
        vector< pair<int,double> >ln;
        adj.push_back(ln);
        rev_adj.push_back(ln);
    }

    for(int i = 0; i<n; ++i){
        string l; double c,a,b;
        fin>>l>>c>>a>>b;
        string new_l; double new_m;
        fmemory>>new_l>>new_m;
        //graph_node cur_layer(l,c,a,b);
        graph_node cur_layer(l,c,new_m-WORKER_MEMORY+20,new_m-WORKER_MEMORY+20);
        vertex_map[l] = i;
        rev_vertex_map[i] = l;
        gg.push_back(cur_layer);
        tot_memory = tot_memory + new_m-WORKER_MEMORY+20.0;
    }

    for(int i = 1; i<=m; ++i){
        string l1,l2; double transfer;
        fin>>l1>>l2>>transfer;
        int u = vertex_map[l1];
        int v = vertex_map[l2];
        adj[u].push_back(make_pair(v,transfer));
        rev_adj[v].push_back(make_pair(u,transfer));
    }
    //cout<<n<<" "<<m<<"\n";
    fin.close();
    fmemory.close();
    cout<<tot_memory<<"\n";
    cout<<"hoise\n";
}

void take_input_dev_graph(){
    ifstream fin2("dev_graph.txt");
    //format
    //device_no
    //device_num, speed, memory...
    //bandwidht with master....
    int num_device;
    fin2>>num_device;
    for(int i = 0; i<num_device; ++i){
        string ip; double sp,mem;
        fin2>>ip>>sp>>mem;
        device_node cur_device(ip,sp,mem-WORKER_MEMORY);
        dd.push_back(cur_device);
        device_map[ip] = i;
        rev_device_map[i] = ip;
    }
    //ofstream fout("tbgap_map_out.txt");

    for(int i = 0; i<num_device; ++i){
        fin2>>BW_matrix_master[i];
    }
    fin2.close();
    dd_copy = dd;
}

int visited[MXLAYER];

int find_stage_value(int cur_node){
    int sz = 1;
    visited[cur_node] = 1;
    for(auto u: rev_adj[cur_node]){
        int v = u.first;
        if(visited[v]==0){
            sz = sz + find_stage_value(v);
        }
    }
    return sz;
}

vector< pair<int,int> >topo_sort(int layer_no){
    cout<<layer_no<<"\n";
    vector< int >indeg; //pair.first = degree, pair.second = stage_val

    for(int i = 0; i<layer_no; ++i)indeg.push_back(0);

    for(int i = 0; i<layer_no; ++i){
        for(int j = 0; j<adj[i].size(); ++j){
            int v = adj[i][j].first;
            indeg[v]++;
        }
    }
    //cout<<"indeg ber hoise\n";
    priority_queue< pair<int,int> >pq;

    for(int i = 0; i<indeg.size(); ++i){
        pq.push(make_pair(-indeg[i],i));
    }

    vector< pair<int,int> >topo;

    //cout<<"topo shuru hoise\n";

    for(int i = 1; i<=layer_no; ++i){
        pair<int,int> cur = pq.top();
        cout<<cur.first<<" "<<cur.second<<"\n";
        if(cur.first!=0){
            throw "Not Acyclic Exception";
        }
        pq.pop();
        int cur_node = cur.second;
        cout<<cur_node<<"\n";
        memset(visited,0,sizeof(visited));
        int stage_value = find_stage_value(cur_node);

        for(auto v: adj[cur_node]){
            int u = v.first;
            indeg[u] = indeg[u] - 1;
            pq.push(make_pair(-indeg[u],u));
        }
        topo.push_back(make_pair(stage_value,cur_node));
    }
    sort(topo.begin(),topo.end());
    return topo;
}

map<int,int>P1;

vector<int>mask_to_vec(long long msk, int num_device, int num_layers){
    vector<int>lst;

    for(int i = 1; i<=num_layers; ++i){
        lst.push_back(msk%num_device);
        msk = msk/num_device;
    }

    return lst;
}

double evaluate(vector<int>cur_stage_layers, vector<int>cur_devices){
    double comp_time = 0.0;
    double comm_time = 0.0;

    vector<double>device_time;
    vector<device_node>my_copy = dd;

    for(int i = 0; i<my_copy.size(); ++i){
        device_time.push_back(0.0);
    }

    for(int i = 0; i<cur_stage_layers.size(); ++i){
        int my_layer = cur_stage_layers[i];
        int my_device = cur_devices[i];

        if(gg[my_layer].alpha_memory>my_copy[my_device].memory){
            return 1e28;
        }

        else{
            my_copy[my_device].memory = my_copy[my_device].memory-gg[my_layer].alpha_memory;
            device_time[my_device] += (gg[my_layer].num_computation)/(dd[my_device].op_per_time);
            for(auto eg: rev_adj[my_layer]){
                int prev_layer = eg.first;
                double tx = eg.second;
                int prev_device = P1[prev_layer];

                if(prev_device!=my_device)comm_time = comm_time+(tx*8.0)/(BW_matrix_master[prev_device]*1024.0)+(tx*8.0)/(BW_matrix_master[my_device]*1024.0);
            }

        }
        //comp_time = comp_time +
    }

    for(double pocha: device_time)comp_time = max(comp_time,pocha);
    return alpha*comp_time+(1.0-alpha)*comm_time;
}

void partial_place(vector<int>cur_stage_layers, vector<int>cur_devices){
    for(int i = 0; i<cur_devices.size(); ++i){
        int my_layer = cur_stage_layers[i];
        int my_device = cur_devices[i];
        cout<<my_device<<" had memory "<<dd[my_device].memory<<"\n";
        dd[my_device].memory = dd[my_device].memory-gg[my_layer].alpha_memory;
        P1[my_layer] = my_device;
        cout<<"placing "<<rev_vertex_map[my_layer]<<" to "<<my_device<<" needing "<<gg[my_layer].alpha_memory<<"memory\n";
        cout<<"now "<<my_device<<" has "<<dd[my_device].memory<<"\n";
    }
}

void find_first_placement(vector<pair<int,int> >topo_stage){
    vector< vector<int> >layer_by_stage;

    int sz = topo_stage.size();

    for(int i = 0; i<sz+3; ++i){
        vector<int>ln;
        layer_by_stage.push_back(ln);
    }

    for(auto it: topo_stage){
        int stg = it.first;
        int vtx = it.second;
        layer_by_stage[stg].push_back(vtx);
    }
    //thik ache mone hocche
    for(int i = 0; i<(int)layer_by_stage.size(); ++i){
        vector<int>cur_layers = layer_by_stage[i];

        long long max_device_mask = 1;
        for(int j = 0; j<cur_layers.size(); ++j){
            max_device_mask = max_device_mask*(int)dd.size();
        }

        vector<int>best_list;
        double best_time = 1e28;

        for(int msk = 0; msk<max_device_mask; ++msk){
            vector<int>lst_device = mask_to_vec(msk,(int)dd.size(),(int)cur_layers.size());

            double cur_time = evaluate(cur_layers,lst_device);

            if((cur_time<best_time)){
                best_time = cur_time;
                best_list = lst_device;
            }
        }
        partial_place(cur_layers,best_list);
   }
}

void ordered_placer(){
    vector< pair<double,int> >device_ordered;
    vector< pair<double,int> >layer_ordered;

    for(int i = 0; i<(int)dd.size(); ++i){
        device_ordered.push_back(make_pair(dd[i].memory,i));
    }
    for(int i = 0; i<(int)gg.size(); ++i){
        layer_ordered.push_back(make_pair(gg[i].alpha_memory,i));
    }

    sort(device_ordered.begin(),device_ordered.end());
    sort(layer_ordered.begin(),layer_ordered.end());

    ofstream f_order("order_mapping.txt");
    int cur_device_idx = device_ordered.size()-1;
    int cur_layer_idx = 0;

    while(cur_layer_idx<(int)layer_ordered.size()){
        int cur_layer = layer_ordered[cur_layer_idx].second;
        int cur_device = device_ordered[cur_device_idx].second;

        if(gg[cur_layer].alpha_memory<=dd[cur_device].memory){
            f_order<<gg[cur_layer].layer_name<<" "<<cur_device<<"\n";
            dd[cur_device].memory = dd[cur_device].memory-gg[cur_layer].alpha_memory;
            cur_layer_idx++;
	    cout<<"placing "<<cur_layer<<" in "<<cur_device<<"\n";
        }
        else{
            cur_device_idx--;
        }
    }
    f_order.close();

}

int main(){
    cout<<"hello\n";
    //cin>>alpha;
    alpha=0.5;
    take_input_ml_graph();
    take_input_dev_graph();
    cout<<gg.size()<<endl;
    vector< pair<int,int> >tp = topo_sort(gg.size());
    for(auto u: tp){
        cout<<u.first<<" "<<u.second<<"\n";
    }
    ordered_placer();
    return 0;
}
