// ==============================================
// Autor: Paganini Barcellos de Oliveira
// ==============================================
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <list>
#include <cmath>
#include <ctime>
#include "lap.h"
#include <lemon/smart_graph.h>
#include <lemon/network_simplex.h>
using namespace lemon;
using namespace std;
#include <ilcplex/ilocplex.h>
#define EPSILON 0.000001
#define LAMBDA 0.5
#define cd0 0.5
#define f0 0.5
#define w0 0.5
#define zero 0.000001
#define PERC_CLOS 0.1
#define PERC_FUNC 0.3
#define percH 0.15
#define ALPHA 21
#define MAXITER 10
ILOSTLBEGIN

using Weight = double;
using Capacity = double;
using Graph = SmartDigraph;
using Node = Graph::Node;
using Arc = Graph::Arc;
template<typename ValueType>
using ArcMap = SmartDigraph::ArcMap<ValueType>;
using NS = NetworkSimplex<SmartDigraph, Capacity, Weight>;

// ==============================================
// Estrutura de dados
// ==============================================
typedef struct{
    int np;                                                   // Qtde de Periodos
    int nc;                                                   // Qtde de Clientes
    int ncd;                                                  // Qtde de CD's
    int nf;                                                   // Qtde de Fabricas
    vector <vector<double> > ccd_open;                        // Custo de instalação de CD's no perido - openning
    vector <vector<double> > ccd_clos;                        // Custo de instalação de CD's no perido - closing
    vector <vector<double> > ccd_func;                        // Custo de instalação de CD's no perido - functioning
    vector <vector<double> > cf_open;                         // Custo de instalação de Fabricas - openning
    vector <vector<double> > cf_clos;                         // Custo de instalação de Fabricas - closing
    vector <vector<double> > cf_func;                         // Custo de instalação de Fabricas - functioning
    vector <vector<double> > d;                               // Demanda de cada cliente
    vector <vector<vector<double> > > c1;                     // Custo de transporte do CD j para o Cliente i
    vector <vector<vector<double> > > c2;                     // Custo de transporte da Fabrica k para o CD j
    vector <vector<vector<vector<double> > > > c;             // Custo de transporte total - open
    char name[250];

    vector <vector <vector<int> > > best_arc;
    vector <vector <vector<int> > > min_arc;
    vector <vector<double> > dif_minCD;
    vector <vector<double> > dif_minF;
    
    vector <double> lb;
    vector <double> sup;
    vector <double> ub;
    vector <double> time;
    
    int typemod;                                    // Model: (1)MM model or (2)MMỹ model
    int hcut;                                       // Heuristic custs? (1) yes (0) No
    int setPrio;                                    // SetPriorities? (1) yes (0) No
    int typehcut;                                   // Type of heuristic custs: (0) None (1) OF cut (2) Inspection Benders cuts (3) Both 1 and 2
    int place;                                      // Where I'll start B&B? (1) heuristic solution (0) CLEX defaut
    int warm;                                       // Warm start? (1) yes (0) No
    int hh;                                         // To define the number of warm start iterations
    int hlr;                                        // Warm start until LR? (1) yes (0) No
    int ccall;                                      // Cutcallback? (1) yes (0) No 
    int ncuts;                                      // To define the number of Cutcallback iterations - ((-1) defaut to insert all of them)
    int icut;                                       // Inspection cuts? (1) yes (0) No 
    int cfcut;                                      // closing facility cuts? (1) yes (0) No 
    int pcut;                                       // Papadakos cuts? (1) yes (0) No 
    int mwcut;                                      // Magnanti Wong cuts? (1) yes (0) No 
    int iJVCcut;                                    // closing facility JVC cuts? (1) yes (0) No 
    int iJVCcfcut;                                  // Inspection JVC cuts? (1) yes (0) No
    int mwpcut;                                     // Magnanti Wong - Pareto cuts? (1) yes (0) No 
       
    int iter;
    IloEnv env;
    IloTimer *crono;
    double time_heu;
    vector <vector <vector<double> > > c_minCusto;
    vector <vector <vector<int> > > c_minF;
    vector <vector <vector<int> > > c_minCD;
    IloNum trans_min;
    
} DAT;

typedef struct{
    IloEnv env;
    IloCplex cplex;
    IloModel mod;
    IloNumVarArray f_open;
    IloNumVarArray f_clos;
    IloNumVarArray f_func;
    IloNumArray _f_open;
    IloNumArray _f_clos;
    IloNumArray _f_func;
    IloNumArray _f_func_CF;
    IloNumVarArray cd_open;
    IloNumVarArray cd_clos;
    IloNumVarArray cd_func;
    IloNumArray _cd_open;
    IloNumArray _cd_clos;
    IloNumArray _cd_func;
    IloNumArray _cd_func_CF;
    IloNumVarArray eta;
    IloRangeArray constraints;
    IloRangeArray cuts;
    IloObjective fo;
    IloNum ub;
    IloNum lb;
    IloNum gap;
    IloNum old_lb;
    IloTimer *crono;
    IloNum valor;
    
    IloNumArray f0_m;
    IloNumArray cd0_m;
    
    //MW lemon
    IloNumArray f0_mw_open;
    IloNumArray cd0_mw_open;
    IloNumArray f0_mw_clos;
    IloNumArray cd0_mw_clos;
    IloNumArray f0_mw_func;
    IloNumArray cd0_mw_func;
    
    //Set variables priorities
    IloNumArray varZ_open;        // If use set priorities 1 level
    IloNumArray varZ_clos;        // If use set priorities 1 level
    IloNumArray varZ_func;        // If use set priorities 1 level
    IloNumArray varY_open;        // If use set priorities 2 level
    IloNumArray varY_clos;        // If use set priorities 2 level
    IloNumArray varY_func;        // If use set priorities 2 level
    
    //Heurística
    IloNumArray v_aux;
    IloNumArray u_aux;
    IloNumArray w_aux;
    IloNum ub_aux;
    
} MP_CPX_DAT;

typedef struct{
    IloEnv env;
    IloCplex cplex;
    IloModel mod;
    IloNumVarArray v;
    IloNumArray _v;
    IloNumVarArray u;
    IloNumArray _u;
    IloNumVarArray w;
    IloNumArray _w;
    IloRangeArray constraints;
    IloObjective fo;
    IloNum of;
    IloNumArray _coef_f_func;
    IloNumArray _coef_cd_func;
    
    //Papadakos
    IloNumArray _f0_open;
    IloNumArray _cd0_open;
    IloNumArray _f0_clos;
    IloNumArray _cd0_clos;
    IloNumArray _f0_func;
    IloNumArray _cd0_func;
} SPD_CPX_DAT;

typedef struct{
    vector<vector<vector<double> > > u;
    vector<vector<vector<double> > > w;
    vector<vector<double> > v;
    double _of;
    
    vector <vector<int> > open_cd;
    vector <vector<int> > open_fab;
    
    double *u_aux;
    double *w_aux;
    double **assigncost;
    int* rowsol;
    int* colsol;
    int dim;
} SPDI_CPX_DAT;

typedef struct{ // Estrutura de dados para a heurística
    vector<vector<vector<int> > > s;                       // Indica a posição do menor custo em cada período de cada cliente
    vector <vector<bool> > cd_open;                        // Instalação de CD's no perido - openning
    vector <vector<bool> > cd_clos;                        // Fechamento de CD's no perido - closing
    vector <vector<bool> > cd_func;                        // Funcionamento de CD's no perido - functioning
    vector <vector<bool> > f_open;                         // Instalação de Fabricas - openning
    vector <vector<bool> > f_clos;                         // Fechamento de Fabricas - closing
    vector <vector<bool> > f_func;                         // Funcionamento de Fabricas - functioning
    
    vector<vector<vector<int> > > s_star;                  // Indica a posição do menor custo em cada período de cada cliente
    vector <vector<bool> > cd_open_star;                   // Instalação de CD's no perido - openning
    vector <vector<bool> > cd_clos_star;                   // Fechamento de CD's no perido - closing
    vector <vector<bool> > cd_func_star;                   // Funcionamento de CD's no perido - functioning
    vector <vector<bool> > f_open_star;                    // Instalação de Fabricas - openning
    vector <vector<bool> > f_clos_star;                    // Fechamento de Fabricas - closing
    vector <vector<bool> > f_func_star;                    // Funcionamento de Fabricas - functioning
    
    vector<vector<vector<int> > > s_starG;                 // Indica a posição do menor custo em cada período de cada cliente
    vector <vector<bool> > cd_open_starG;                  // Instalação de CD's no perido - openning
    vector <vector<bool> > cd_clos_starG;                  // Fechamento de CD's no perido - closing
    vector <vector<bool> > cd_func_starG;                  // Funcionamento de CD's no perido - functioning
    vector <vector<bool> > f_open_starG;                   // Instalação de Fabricas - openning
    vector <vector<bool> > f_clos_starG;                   // Fechamento de Fabricas - closing
    vector <vector<bool> > f_func_starG;                   // Funcionamento de Fabricas - functioning
    
    double fo;
    double fo_star;
    double fo_starG;
    IloNum trans_s;                                             //custo de transporte da solução atual;
    IloNum instC_s;                                             //custo de instalação da solução atual;
    IloNum trans_s_star;                                        //custo de transporte da solução atual;
    IloNum instC_s_star;                                        //custo de instalação da solução atual;
    IloNum trans_s_starG;                                       //custo de transporte da solução atual;
    IloNum instC_s_starG;                                       //custo de instalação da solução atual;
    
    //Parâmetros para a função "F1"
    vector <double> vc;
    vector <vector<int> > a;
    double minF1;
    double maxF1;
    
    double alpha;                                                // Fator de aleatoriedade do método
    vector <vector<int> > open_cd;
    vector <vector<int> > open_fab;
    
    //Cópias
    vector<vector<vector<int> > > s_copia;                       // Indica a posição do menor custo em cada período de cada cliente
    vector <vector<bool> > cd_open_copia;                        // Instalação de CD's no perido - openning
    vector <vector<bool> > cd_clos_copia;                        // Fechamento de CD's no perido - closing
    vector <vector<bool> > cd_func_copia;                        // Funcionamento de CD's no perido - functioning
    vector <vector<bool> > f_open_copia;                         // Instalação de Fabricas - openning
    vector <vector<bool> > f_clos_copia;                         // Fechamento de Fabricas - closing
    vector <vector<bool> > f_func_copia;                         // Funcionamento de Fabricas - functioning
    double fo_copia;
    
    //JVC
    double *u_aux;
    double *w_aux;
    double **assigncost;
    int* rowsol;
    int* colsol;
    int dim;
    
    //demandas
    vector <int> demand;
    vector <double> media_demand;
    int choose; //if h.choose = 1 -> open, if h.choose = 2 -> close
    
} HEU_DAT;

// ==============================================
// Funções auxiliares
// ==============================================
void read_data(char name[],DAT &d);
void read_data_CF(DAT &d);
void help();
void create_model_mp(DAT &d, MP_CPX_DAT &mp, HEU_DAT &h);                  
void create_model_spd(DAT &d, MP_CPX_DAT &mp, SPD_CPX_DAT &spd);           
void create_model_spdi(DAT &d, SPDI_CPX_DAT &spdi);                        
void solve_mp(DAT &d, MP_CPX_DAT &mp);
double solve_spd(DAT &d, MP_CPX_DAT &mp, SPD_CPX_DAT &spd);
double solve_papadakos(DAT &d, MP_CPX_DAT &mp, SPD_CPX_DAT &spd);
void solve_spdiMW(DAT &d, MP_CPX_DAT &mp, SPDI_CPX_DAT &spdi);
void solve_spdiMWPareto(DAT &d, MP_CPX_DAT &mp, SPDI_CPX_DAT &spdi);
void add_n_cuts(DAT &d, MP_CPX_DAT &mp, SPD_CPX_DAT &spd);
double solve_spdi(DAT &d, MP_CPX_DAT &mp, SPDI_CPX_DAT &spdi);
void create_vector_spdi(DAT &d, MP_CPX_DAT &mp, SPDI_CPX_DAT &spdi);
double solve_spdiJVC(DAT &d, MP_CPX_DAT &mp, SPDI_CPX_DAT &spdi);
void print_solution(DAT &d, MP_CPX_DAT &mp);
void print_solution_new(DAT &d, MP_CPX_DAT &mp);
void print_solution_arq(DAT &d, HEU_DAT &h, MP_CPX_DAT &mp, char *argv[], double t_final);
int valor_min(DAT &d, SPDI_CPX_DAT &spdi, int i, int t);
void aux_CF(DAT &d, SPDI_CPX_DAT &spdi, int i, int t);
void min_cost_flow(DAT &d, MP_CPX_DAT &mp, SPDI_CPX_DAT &spdi, int t, int i, double demanda);

void read_data_heu(DAT &d, HEU_DAT &h);
void quicksort_mincost(DAT &d);
void quicksort(DAT &d, int inicio_global, int fim_global, int i, int t);
void solucao_inicial(DAT &d, HEU_DAT &h, int iter);
void define_solucao(DAT &d, HEU_DAT &h);
double calcula_fo_global(DAT &d, HEU_DAT &h);
void update_s_star(DAT &d, HEU_DAT &h);
void F1(DAT &d, HEU_DAT &h);
void quickF1(HEU_DAT &h, int inicio_global, int fim_global);
void grasp(DAT &d, HEU_DAT &h);
void open_clos_grasp(DAT &d, HEU_DAT &h, int tt);
int open_or_close_facility(DAT &d, HEU_DAT &h, int level, int tt);
int verifica_sol(DAT &d, HEU_DAT &h, int tt);
void copia_sol(DAT &d, HEU_DAT &h);
void recover_sol(DAT &d, HEU_DAT &h);
void quickCurrent(double *c, int *a, int inicio_global, int fim_global);
void define_rand_solution(DAT &d, HEU_DAT &h, int level, int tamanho, double *c, int *a, int tt);
void quickCurrentFabCD(double *c, int *a, int *b, int inicio_global, int fim_global);
void define_rand_solutionFabCD(DAT &d, HEU_DAT &h, int level, int tamanho, double *c, int *a, int *b, int tt);
void print_solution_heu(DAT &d, HEU_DAT &h);
void solve_spdiJVC_heu(DAT &d, MP_CPX_DAT &mp, HEU_DAT &h);
void solve_spdi_heu(DAT &d, MP_CPX_DAT &mp, HEU_DAT &h);
void create_vector(DAT &d, HEU_DAT &h);
void update_s(DAT &d, HEU_DAT &h);
void print_solution_heu_current(DAT &d, HEU_DAT &h);

int typedemand(DAT &d, HEU_DAT &h);
void try_changeS(DAT &d, HEU_DAT &h);
void closed_front_demand(DAT &d, HEU_DAT &h, int tt);
int closed_front(DAT &d, HEU_DAT &h, int level, int tt);
void define_front_clos(DAT &d, HEU_DAT &h, int level, int tamanho, double *c, int *a, int tt);
void close_end_to_start(DAT &d, HEU_DAT &h);
void closed_back_demand(DAT &d, HEU_DAT &h, int tt);
int closed_back(DAT &d, HEU_DAT &h, int level, int tt);
void define_back_clos(DAT &d, HEU_DAT &h, int level, int tamanho, double *c, int *a, int tt);
void zera_sol(DAT &d, HEU_DAT &h);
void update_s_starG(DAT &d, HEU_DAT &h);
int close_test(DAT &d, HEU_DAT &h);

// ==============================================
// Funções CALLBACK
// ==============================================
ILOMIPINFOCALLBACK4(infoCallback, DAT &, d, MP_CPX_DAT &, mp, SPD_CPX_DAT &, spd, SPDI_CPX_DAT &, spdi){
    if (hasIncumbent() == IloTrue){
        d.sup.push_back(getIncumbentObjValue());
        d.lb.push_back(getBestObjValue());
        double tot = mp.crono->getTime(); 
        d.time.push_back(tot);
    }
}

// ==============================================
// Função CALLBACK 2
// ==============================================
ILOLAZYCONSTRAINTCALLBACK4(cutCallback, DAT &, d, MP_CPX_DAT &, mp, SPD_CPX_DAT &, spd, SPDI_CPX_DAT &, spdi){

    if (d.typemod == 1){
        getValues(mp._f_func, mp.f_func);
        getValues(mp._cd_func, mp.cd_func);
    }
    else{
        getValues(mp._f_open, mp.f_open);
        getValues(mp._f_clos, mp.f_clos);
        getValues(mp._cd_open, mp.cd_open);
        getValues(mp._cd_clos, mp.cd_clos);
    }
    
    if (d.icut == 1 || d.cfcut == 1){
        solve_spdi(d, mp, spdi);
        
        if (d.icut == 1){
            
            for (int t = 1; t <= d.np; t++){
                for (int i = 1; i <= d.nc; i++){
                    IloExpr cut_insp(getEnv());
                    cut_insp += mp.eta[d.nc * (t - 1) + i - 1];
                    cut_insp -= spdi.v[t][i];
                    if (d.typemod == 1){
                        for (int k = 1; k <= d.nf; k++) cut_insp += spdi.u[t][i][k] * mp.f_func[d.nf * (t - 1) + k - 1];
                        for (int j = 1; j <= d.ncd; j++) cut_insp += spdi.w[t][i][j] * mp.cd_func[d.ncd * (t - 1) + j - 1];
                    }
                    else{
                        for (int k = 1; k <= d.nf; k++){
                            for (int r = 1; r <= t; r++) cut_insp += spdi.u[t][i][k] * mp.f_open[d.nf * (r - 1) + k - 1];
                            for (int r = 2; r <= t; r++) cut_insp -= spdi.u[t][i][k] * mp.f_clos[d.nf * (r - 1) + k - 1];
                        }
                        for (int j = 1; j <= d.ncd; j++){
                            for (int r = 1; r <= t; r++) cut_insp += spdi.w[t][i][j] * mp.cd_open[d.ncd * (r - 1) + j - 1];
                            for (int r = 2; r <= t; r++) cut_insp -= spdi.w[t][i][j] * mp.cd_clos[d.ncd * (r - 1) + j - 1];
                        }
                    }
                    add(cut_insp >= 0);
                    cut_insp.end();
                }
            }
        }
        
        if (d.cfcut == 1){
            for (int t = 1; t <= d.np; t++){
                for (int i = 1; i <= d.nc; i++){
                    IloExpr cut_insp(getEnv());
                    cut_insp += mp.eta[d.nc * (t - 1) + i - 1];
                    cut_insp -= spdi.v[t][i];
                    if (d.typemod == 1){
                        for (int k = 1; k <= d.nf; k++) cut_insp += spdi.u[t][i][k] * mp.f_func[d.nf * (t - 1) + k - 1];
                        for (int j = 1; j <= d.ncd; j++) cut_insp += spdi.w[t][i][j] * mp.cd_func[d.ncd * (t - 1) + j - 1];
                    }
                    else{
                        double val = 0;
                        for (int k = 1; k <= d.nf; k++){
                            for (int r = 1; r <= t; r++){
                                cut_insp += spdi.u[t][i][k] * mp.f_open[d.nf * (r - 1) + k - 1];
                                val += mp._f_open[d.nf * (r - 1) + k - 1];
                            }
                            for (int r = 2; r <= t; r++){
                                cut_insp -= spdi.u[t][i][k] * mp.f_clos[d.nf * (r - 1) + k - 1];
                                val -= mp._f_clos[d.nf * (r - 1) + k - 1];
                            }
                            if(val < 0.9) mp._f_func_CF[d.nf * (t - 1) + k - 1] = 0;
                            else mp._f_func_CF[d.nf * (t - 1) + k - 1] = 1;
                            val = 0;
                        }
                        val = 0;
                        for (int j = 1; j <= d.ncd; j++){
                            for (int r = 1; r <= t; r++){
                                cut_insp += spdi.w[t][i][j] * mp.cd_open[d.ncd * (r - 1) + j - 1];
                                val += mp._cd_open[d.ncd * (r - 1) + j - 1];
                            }
                            for (int r = 2; r <= t; r++){
                                cut_insp -= spdi.w[t][i][j] * mp.cd_clos[d.ncd * (r - 1) + j - 1];
                                val -= mp._cd_clos[d.ncd * (r - 1) + j - 1];
                            }
                            if(val < 0.9) mp._cd_func_CF[d.ncd * (t - 1) + j - 1] = 0;
                            else mp._cd_func_CF[d.ncd * (t - 1) + j - 1] = 1;
                            val = 0;
                        }
                    }
                    
                    aux_CF(d, spdi, i, t);
                    
                    double total = 0;
                    if (d.typemod == 1){
                        for(int j = 1; j <= d.ncd; j++){
                            if(mp._cd_func[d.ncd * (t - 1) + j - 1] > 0.90){
                                if(d.min_arc[t][i][1] == j) total = total + d.dif_minCD[t][i];
                                cut_insp += (1- mp.cd_func[d.ncd * (t - 1) + j - 1]) * total;
                                total = 0;
                            }
                        }
                        total = 0;
                        for(int k = 1; k <= d.nf; k++){
                            if(mp._f_func[d.nf * (t - 1) + k - 1] > 0.90){
                                if(d.min_arc[t][i][2] == k) total = total + d.dif_minF[t][i];
                                cut_insp += (1- mp.f_func[d.nf * (t - 1) + k - 1]) * total;
                                total = 0;
                            }
                        }     
                    }
                    else{
                        for(int j = 1; j <= d.ncd; j++){
                            if(mp._cd_func_CF[d.ncd * (t - 1) + j - 1] > 0.90){
                                if(d.min_arc[t][i][1] == j) total = total + d.dif_minCD[t][i];
                                cut_insp += total;
                                for (int r = 1; r <= t; r++) cut_insp -= total * mp.cd_open[d.ncd * (r - 1) + j - 1];
                                for (int r = 2; r <= t; r++) cut_insp += total * mp.cd_clos[d.ncd * (r - 1) + j - 1];
                                total = 0;
                            }
                        }
                        total = 0;
                        for(int k = 1; k <= d.nf; k++){
                            if(mp._f_func_CF[d.nf * (t - 1) + k - 1] > 0.90){
                                if(d.min_arc[t][i][2] == k) total = total + d.dif_minF[t][i];
                                cut_insp += total;
                                for (int r = 1; r <= t; r++) cut_insp -= total * mp.f_open[d.nf * (r - 1) + k - 1];
                                for (int r = 2; r <= t; r++) cut_insp += total * mp.f_clos[d.nf * (r - 1) + k - 1];
                                total = 0;
                            }
                        }     
                    }
                    add(cut_insp >= 0);
                    cut_insp.end();
                }
            }

        }
    }
    
    if (d.pcut == 1){
        
        for (int t = 1; t <= d.np; t++){
            if (d.typemod == 1){
                for (int k = 1; k <= d.nf; k++) spd._f0_func[d.nf * (t - 1) + k - 1] = spd._f0_func[d.nf * (t - 1) + k - 1]*(1-LAMBDA) + mp._f_func[d.nf * (t - 1) + k - 1]*LAMBDA;
                for (int j = 1; j <= d.ncd; j++) spd._cd0_func[d.ncd * (t - 1) + j - 1] = spd._cd0_func[d.ncd * (t - 1) + j - 1]*(1-LAMBDA) + mp._cd_func[d.ncd * (t - 1) + j - 1]*LAMBDA;
            }
            else{
                for (int k = 1; k <= d.nf; k++){
                    spd._f0_open[d.nf * (t - 1) + k - 1] = spd._f0_open[d.nf * (t - 1) + k - 1]*(1-LAMBDA) + mp._f_open[d.nf * (t - 1) + k - 1]*LAMBDA;
                    spd._f0_clos[d.nf * (t - 1) + k - 1] = spd._f0_clos[d.nf * (t - 1) + k - 1]*(1-LAMBDA) + mp._f_clos[d.nf * (t - 1) + k - 1]*LAMBDA;
                }
                for (int j = 1; j <= d.ncd; j++){
                    spd._cd0_open[d.ncd * (t - 1) + j - 1] = spd._cd0_open[d.ncd * (t - 1) + j - 1]*(1-LAMBDA) + mp._cd_open[d.ncd * (t - 1) + j - 1]*LAMBDA;
                    spd._cd0_clos[d.ncd * (t - 1) + j - 1] = spd._cd0_clos[d.ncd * (t - 1) + j - 1]*(1-LAMBDA) + mp._cd_clos[d.ncd * (t - 1) + j - 1]*LAMBDA;
                }
            }
        }

        solve_papadakos(d, mp, spd);
        
        for (int t = 1; t <= d.np; t++){
            for (int i = 1; i <= d.nc; i++){
                IloExpr cut_papadakos(getEnv());
                cut_papadakos += mp.eta[d.nc * (t - 1) + i - 1];
                cut_papadakos -= spd._v[d.nc * (t - 1) + i - 1];
                if (d.typemod == 1){
                    for (int k = 1; k <= d.nf; k++) cut_papadakos += spd._u[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] * mp.f_func[d.nf * (t - 1) + k - 1];
                    for (int j = 1; j <= d.ncd; j++) cut_papadakos += spd._w[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] * mp.cd_func[d.ncd * (t - 1) + j - 1];
                }
                else{
                    for (int k = 1; k <= d.nf; k++){
                        for (int r = 1; r <= t; r++) cut_papadakos += spd._u[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] * mp.f_open[d.nf * (r - 1) + k - 1];
                        for (int r = 2; r <= t; r++) cut_papadakos -= spd._u[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] * mp.f_clos[d.nf * (r - 1) + k - 1];
                    }
                    for (int j = 1; j <= d.ncd; j++){
                        for (int r = 1; r <= t; r++) cut_papadakos += spd._w[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] * mp.cd_open[d.ncd * (r - 1) + j - 1];
                        for (int r = 2; r <= t; r++) cut_papadakos -= spd._w[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] * mp.cd_clos[d.ncd * (r - 1) + j - 1];
                    }
                }
                add(cut_papadakos >= 0);
                cut_papadakos.end();
            }
        }
    }
    
    if (d.mwcut == 1){
        
        solve_spdiMW(d, mp, spdi);
        
        for (int t = 1; t <= d.np; t++){
            for (int i = 1; i <= d.nc; i++){
                IloExpr cut_mw(getEnv());
                cut_mw += mp.eta[d.nc * (t - 1) + i - 1];
                cut_mw -= spdi.v[t][i];
                if (d.typemod == 1){
                    for (int k = 1; k <= d.nf; k++) cut_mw += spdi.u[t][i][k] * mp.f_func[d.nf * (t - 1) + k - 1];
                    for (int j = 1; j <= d.ncd; j++) cut_mw += spdi.w[t][i][j] * mp.cd_func[d.ncd * (t - 1) + j - 1];
                }
                else{
                    for (int k = 1; k <= d.nf; k++){
                        for (int r = 1; r <= t; r++) cut_mw += spdi.u[t][i][k] * mp.f_open[d.nf * (r - 1) + k - 1];
                        for (int r = 2; r <= t; r++) cut_mw -= spdi.u[t][i][k] * mp.f_clos[d.nf * (r - 1) + k - 1];
                    }
                    for (int j = 1; j <= d.ncd; j++){
                        for (int r = 1; r <= t; r++) cut_mw += spdi.w[t][i][j] * mp.cd_open[d.ncd * (r - 1) + j - 1];
                        for (int r = 2; r <= t; r++) cut_mw -= spdi.w[t][i][j] * mp.cd_clos[d.ncd * (r - 1) + j - 1];
                    }
                }
                add(cut_mw >= 0);
                cut_mw.end();
            }
        }
    }
    
    if (d.iJVCcut == 1 || d.iJVCcfcut == 1){
        solve_spdiJVC(d, mp, spdi);
        
        if (d.iJVCcut == 1){
            for (int t = 1; t <= d.np; t++){
                for (int i = 1; i <= d.nc; i++){
                    IloExpr cut_insp(getEnv());
                    cut_insp += mp.eta[d.nc * (t - 1) + i - 1];
                    cut_insp -= spdi.v[t][i];
                    if (d.typemod == 1){
                        for (int k = 1; k <= d.nf; k++) cut_insp += spdi.u[t][i][k] * mp.f_func[d.nf * (t - 1) + k - 1];
                        for (int j = 1; j <= d.ncd; j++) cut_insp += spdi.w[t][i][j] * mp.cd_func[d.ncd * (t - 1) + j - 1];
                    }
                    else{
                        for (int k = 1; k <= d.nf; k++){
                            for (int r = 1; r <= t; r++) cut_insp += spdi.u[t][i][k] * mp.f_open[d.nf * (r - 1) + k - 1];
                            for (int r = 2; r <= t; r++) cut_insp -= spdi.u[t][i][k] * mp.f_clos[d.nf * (r - 1) + k - 1];
                        }
                        for (int j = 1; j <= d.ncd; j++){
                            for (int r = 1; r <= t; r++) cut_insp += spdi.w[t][i][j] * mp.cd_open[d.ncd * (r - 1) + j - 1];
                            for (int r = 2; r <= t; r++) cut_insp -= spdi.w[t][i][j] * mp.cd_clos[d.ncd * (r - 1) + j - 1];
                        }
                    }
                    add(cut_insp >= 0);
                    cut_insp.end();
                }
            }
        }
        
        if (d.iJVCcfcut == 1){
        
            for (int t = 1; t <= d.np; t++){
                for (int i = 1; i <= d.nc; i++){
                    IloExpr cut_insp(getEnv());
                    cut_insp += mp.eta[d.nc * (t - 1) + i - 1];
                    cut_insp -= spdi.v[t][i];
                    if (d.typemod == 1){
                        for (int k = 1; k <= d.nf; k++) cut_insp += spdi.u[t][i][k] * mp.f_func[d.nf * (t - 1) + k - 1];
                        for (int j = 1; j <= d.ncd; j++) cut_insp += spdi.w[t][i][j] * mp.cd_func[d.ncd * (t - 1) + j - 1];
                    }
                    else{
                        double val = 0;
                        for (int k = 1; k <= d.nf; k++){
                            for (int r = 1; r <= t; r++){
                                cut_insp += spdi.u[t][i][k] * mp.f_open[d.nf * (r - 1) + k - 1];
                                val += mp._f_open[d.nf * (r - 1) + k - 1];
                            }
                            for (int r = 2; r <= t; r++){
                                cut_insp -= spdi.u[t][i][k] * mp.f_clos[d.nf * (r - 1) + k - 1];
                                val -= mp._f_clos[d.nf * (r - 1) + k - 1];
                            }
                            if(val < 0.9) mp._f_func_CF[d.nf * (t - 1) + k - 1] = 0;
                            else mp._f_func_CF[d.nf * (t - 1) + k - 1] = 1;
                            val = 0;
                        }
                        val = 0;
                        for (int j = 1; j <= d.ncd; j++){
                            for (int r = 1; r <= t; r++){
                                cut_insp += spdi.w[t][i][j] * mp.cd_open[d.ncd * (r - 1) + j - 1];
                                val += mp._cd_open[d.ncd * (r - 1) + j - 1];
                            }
                            for (int r = 2; r <= t; r++){
                                cut_insp -= spdi.w[t][i][j] * mp.cd_clos[d.ncd * (r - 1) + j - 1];
                                val -= mp._cd_clos[d.ncd * (r - 1) + j - 1];
                            }
                            if(val < 0.9) mp._cd_func_CF[d.ncd * (t - 1) + j - 1] = 0;
                            else mp._cd_func_CF[d.ncd * (t - 1) + j - 1] = 1;
                            val = 0;
                        }
                    }
                    
                    aux_CF(d, spdi, i, t);
                    
                    double total = 0;
                    if (d.typemod == 1){
                        for(int j = 1; j <= d.ncd; j++){
                            if(mp._cd_func[d.ncd * (t - 1) + j - 1] > 0.90){
                                if(d.min_arc[t][i][1] == j) total = total + d.dif_minCD[t][i];
                                cut_insp += (1- mp.cd_func[d.ncd * (t - 1) + j - 1]) * total;
                                total = 0;
                            }
                        }
                        total = 0;
                        for(int k = 1; k <= d.nf; k++){
                            if(mp._f_func[d.nf * (t - 1) + k - 1] > 0.90){
                                if(d.min_arc[t][i][2] == k) total = total + d.dif_minF[t][i];
                                cut_insp += (1- mp.f_func[d.nf * (t - 1) + k - 1]) * total;
                                total = 0;
                            }
                        }     
                    }
                    else{
                        for(int j = 1; j <= d.ncd; j++){
                            if(mp._cd_func_CF[d.ncd * (t - 1) + j - 1] > 0.90){
                                if(d.min_arc[t][i][1] == j) total = total + d.dif_minCD[t][i];
                                cut_insp += total;
                                for (int r = 1; r <= t; r++) cut_insp -= total * mp.cd_open[d.ncd * (r - 1) + j - 1];
                                for (int r = 2; r <= t; r++) cut_insp += total * mp.cd_clos[d.ncd * (r - 1) + j - 1];
                                total = 0;
                            }
                        }
                        total = 0;
                        for(int k = 1; k <= d.nf; k++){
                            if(mp._f_func_CF[d.nf * (t - 1) + k - 1] > 0.90){
                                if(d.min_arc[t][i][2] == k) total = total + d.dif_minF[t][i];
                                cut_insp += total;
                                for (int r = 1; r <= t; r++) cut_insp -= total * mp.f_open[d.nf * (r - 1) + k - 1];
                                for (int r = 2; r <= t; r++) cut_insp += total * mp.f_clos[d.nf * (r - 1) + k - 1];
                                total = 0;
                            }
                        }     
                    }
                    add(cut_insp >= 0);
                    cut_insp.end();
                }
            }
        }
    }
    
    if (d.mwpcut == 1){
        
        for (int t = 1; t <= d.np; t++){
            if (d.typemod == 1){
                for (int k = 1; k <= d.nf; k++) mp.f0_mw_func[d.nf * (t - 1) + k - 1] = mp.f0_mw_func[d.nf * (t - 1) + k - 1]*(1-LAMBDA) + mp._f_func[d.nf * (t - 1) + k - 1]*LAMBDA;
                for (int j = 1; j <= d.ncd; j++) mp.cd0_mw_func[d.ncd * (t - 1) + j - 1] = mp.cd0_mw_func[d.ncd * (t - 1) + j - 1]*(1-LAMBDA) + mp._cd_func[d.ncd * (t - 1) + j - 1]*LAMBDA;
            }
            else{
                for (int k = 1; k <= d.nf; k++){
                    mp.f0_mw_open[d.nf * (t - 1) + k - 1] = mp.f0_mw_open[d.nf * (t - 1) + k - 1]*(1-LAMBDA) + mp._f_open[d.nf * (t - 1) + k - 1]*LAMBDA;
                    mp.f0_mw_clos[d.nf * (t - 1) + k - 1] = mp.f0_mw_clos[d.nf * (t - 1) + k - 1]*(1-LAMBDA) + mp._f_clos[d.nf * (t - 1) + k - 1]*LAMBDA;
                }
                for (int j = 1; j <= d.ncd; j++){
                    mp.cd0_mw_open[d.ncd * (t - 1) + j - 1] = mp.cd0_mw_open[d.ncd * (t - 1) + j - 1]*(1-LAMBDA) + mp._cd_open[d.ncd * (t - 1) + j - 1]*LAMBDA;
                    mp.cd0_mw_clos[d.ncd * (t - 1) + j - 1] = mp.cd0_mw_clos[d.ncd * (t - 1) + j - 1]*(1-LAMBDA) + mp._cd_clos[d.ncd * (t - 1) + j - 1]*LAMBDA;
                }
            }
        }
                
        solve_spdiMWPareto(d, mp, spdi);
        
        for (int t = 1; t <= d.np; t++){
            for (int i = 1; i <= d.nc; i++){
                IloExpr cut_mw(getEnv());
                cut_mw += mp.eta[d.nc * (t - 1) + i - 1];
                cut_mw -= spdi.v[t][i];
                if (d.typemod == 1){
                    for (int k = 1; k <= d.nf; k++) cut_mw += spdi.u[t][i][k] * mp.f_func[d.nf * (t - 1) + k - 1];
                    for (int j = 1; j <= d.ncd; j++) cut_mw += spdi.w[t][i][j] * mp.cd_func[d.ncd * (t - 1) + j - 1];
                }
                else{
                    for (int k = 1; k <= d.nf; k++){
                        for (int r = 1; r <= t; r++) cut_mw += spdi.u[t][i][k] * mp.f_open[d.nf * (r - 1) + k - 1];
                        for (int r = 2; r <= t; r++) cut_mw -= spdi.u[t][i][k] * mp.f_clos[d.nf * (r - 1) + k - 1];
                    }
                    for (int j = 1; j <= d.ncd; j++){
                        for (int r = 1; r <= t; r++) cut_mw += spdi.w[t][i][j] * mp.cd_open[d.ncd * (r - 1) + j - 1];
                        for (int r = 2; r <= t; r++) cut_mw -= spdi.w[t][i][j] * mp.cd_clos[d.ncd * (r - 1) + j - 1];
                    }
                }
                add(cut_mw >= 0);
                cut_mw.end();
            }
        }
    }
    
}

//Cutcallback Function 
IloCplex::MIPCallbackI::NodeId oldid;
int insere = 0;
int old = oldid._id;
ILOUSERCUTCALLBACK3(cutCallback2, DAT &, d, MP_CPX_DAT &, mp, SPD_CPX_DAT &, spd){
    IloCplex::MIPCallbackI::NodeId cid = getNodeId();
    if (cid._id < old){
        if (d.ncuts > 0) insere++;
        else insere = -2;
	    if (insere < d.ncuts){
            
            if (d.typemod == 1){
                getValues(mp._f_func, mp.f_func);
                getValues(mp._cd_func, mp.cd_func);
            }
            else{
                getValues(mp._f_open, mp.f_open);
                getValues(mp._f_clos, mp.f_clos);
                getValues(mp._cd_open, mp.cd_open);
                getValues(mp._cd_clos, mp.cd_clos);
            }

            
            for (int t = 1; t <= d.np; t++){
                if (d.typemod == 1){
                    for (int k = 1; k <= d.nf; k++) spd._f0_func[d.nf * (t - 1) + k - 1] = spd._f0_func[d.nf * (t - 1) + k - 1]*(1-LAMBDA) + mp._f_func[d.nf * (t - 1) + k - 1]*LAMBDA;
                    for (int j = 1; j <= d.ncd; j++) spd._cd0_func[d.ncd * (t - 1) + j - 1] = spd._cd0_func[d.ncd * (t - 1) + j - 1]*(1-LAMBDA) + mp._cd_func[d.ncd * (t - 1) + j - 1]*LAMBDA;
                }
                else{
                    for (int k = 1; k <= d.nf; k++){
                        spd._f0_open[d.nf * (t - 1) + k - 1] = spd._f0_open[d.nf * (t - 1) + k - 1]*(1-LAMBDA) + mp._f_open[d.nf * (t - 1) + k - 1]*LAMBDA;
                        spd._f0_clos[d.nf * (t - 1) + k - 1] = spd._f0_clos[d.nf * (t - 1) + k - 1]*(1-LAMBDA) + mp._f_clos[d.nf * (t - 1) + k - 1]*LAMBDA;
                    }
                    for (int j = 1; j <= d.ncd; j++){
                        spd._cd0_open[d.ncd * (t - 1) + j - 1] = spd._cd0_open[d.ncd * (t - 1) + j - 1]*(1-LAMBDA) + mp._cd_open[d.ncd * (t - 1) + j - 1]*LAMBDA;
                        spd._cd0_clos[d.ncd * (t - 1) + j - 1] = spd._cd0_clos[d.ncd * (t - 1) + j - 1]*(1-LAMBDA) + mp._cd_clos[d.ncd * (t - 1) + j - 1]*LAMBDA;
                    }
                }
            }
            solve_papadakos(d, mp, spd);
            for (int t = 1; t <= d.np; t++){
                for (int i = 1; i <= d.nc; i++){
                    IloExpr cut_papadakos(getEnv());
                    cut_papadakos += mp.eta[d.nc * (t - 1) + i - 1];
                    cut_papadakos -= spd._v[d.nc * (t - 1) + i - 1];
                    if (d.typemod == 1){
                        for (int k = 1; k <= d.nf; k++) cut_papadakos += spd._u[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] * mp.f_func[d.nf * (t - 1) + k - 1];
                        for (int j = 1; j <= d.ncd; j++) cut_papadakos += spd._w[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] * mp.cd_func[d.ncd * (t - 1) + j - 1];
                    }
                    else{
                        for (int k = 1; k <= d.nf; k++){
                            for (int r = 1; r <= t; r++) cut_papadakos += spd._u[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] * mp.f_open[d.nf * (r - 1) + k - 1];
                            for (int r = 2; r <= t; r++) cut_papadakos -= spd._u[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] * mp.f_clos[d.nf * (r - 1) + k - 1];
                        }
                        for (int j = 1; j <= d.ncd; j++){
                            for (int r = 1; r <= t; r++) cut_papadakos += spd._w[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] * mp.cd_open[d.ncd * (r - 1) + j - 1];
                            for (int r = 2; r <= t; r++) cut_papadakos -= spd._w[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] * mp.cd_clos[d.ncd * (r - 1) + j - 1];
                        }
                    }
                    add(cut_papadakos >= 0);
                    cut_papadakos.end();
                }
            }
        }
    }   
    old = cid._id;
}

// ============================================================================================================================================
// Função Principal
// ============================================================================================================================================
int main (int argc, char *argv[]){
    DAT d;
    d.typemod = (argc > 2) ? atoi(argv[2]) : 1;         // Model: (1)MM model or (2)MMỹ model
    d.setPrio = (argc > 3) ? atoi(argv[3]) : 0;         // SetPriorities? (1) yes (0) No
    d.hcut = (argc > 4) ? atoi(argv[4]) : 0;            // Heuristic custs? (1) yes (0) No
    d.typehcut = (argc > 5) ? atoi(argv[5]) : 0;        // Type of heuristic custs: (0) None (1) OF cut (2) Inspection Benders cuts (3) Both 1 and 2
    d.place = (argc > 6) ? atoi(argv[6]) : 0;           // Where I'll start B&B? (1) heuristic solution (0) CLEX defaut
    d.warm = (argc > 7) ? atoi(argv[7]) : 0;            // Warm start? (1) yes (0) No
    d.hh = (argc > 8) ? atoi(argv[8]) : 0;              // To define the number of warm start iterations
    d.hlr = (argc > 9) ? atoi(argv[9]) : 0;             // Warm start until LR? (1) yes (0) No
    d.ccall = (argc > 10) ? atoi(argv[10]) : 0;         // Cutcallback? (1) yes (0) No 
    d.ncuts = (argc > 11) ? atoi(argv[11]) : -1;        // To define the number of Cutcallback iterations - ((-1) defaut to insert all of them)
    d.icut = (argc > 12) ? atoi(argv[12]) : 1;          // Inspection cuts? (1) yes (0) No 
    d.cfcut = (argc > 13) ? atoi(argv[13]) : 0;         // closing facility cuts? (1) yes (0) No 
    d.pcut = (argc > 14) ? atoi(argv[14]) : 0;          // Papadakos cuts? (1) yes (0) No 
    d.mwcut = (argc > 15) ? atoi(argv[15]) : 0;         // Magnanti Wong cuts? (1) yes (0) No 
    d.iJVCcut = (argc > 16) ? atoi(argv[16]) : 0;       // closing facility JVC cuts? (1) yes (0) No 
    d.iJVCcfcut = (argc > 17) ? atoi(argv[17]) : 0;     // Inspection JVC cuts? (1) yes (0) No 
    d.mwpcut = (argc > 18) ? atoi(argv[18]) : 0;        // Magnanti Wong - Pareto cuts? (1) yes (0) No 

    read_data(argv[1],d);                                 // Leitura dos dados
    // ===============
    // Ambiente cplex
    // ===============
    MP_CPX_DAT mp;
    SPD_CPX_DAT spd;
    SPDI_CPX_DAT spdi;
    HEU_DAT h;
  
    quicksort_mincost(d);
    
    if(d.cfcut == 1 || d.iJVCcfcut == 1) read_data_CF(d);
    
    d.time_heu = 0;
    if(d.hcut == 1){
        //srand(time(NULL));
        d.iter = MAXITER;
        IloTimer crono(d.env);
        d.crono = &crono;
        crono.start();
        read_data_heu(d,h);
        grasp(d, h);
        d.time_heu = crono.getTime();
    }
    
    try {
        IloTimer crono(mp.env);
        mp.crono = &crono;
        
        create_model_mp(d, mp, h);
        create_model_spd(d, mp, spd);
        create_model_spdi(d, spdi);
        
        bool stop = false;
        int ih = 0;
        mp.ub = IloInfinity;
        mp.old_lb = -IloInfinity;
        IloNum delta_lb;
        
        IloEnv env = spd.cplex.getEnv();
        if (d.typemod == 1){
            spd._f0_func = IloNumArray(env, d.np * d.nf);
            spd._cd0_func = IloNumArray(env, d.np * d.ncd);
            for (int t = 1; t <= d.np; t++){
                for (int k = 1; k <= d.nf; k++) spd._f0_func[d.nf * (t - 1) + k - 1] = f0;
                for (int j = 1; j <= d.ncd; j++) spd._cd0_func[d.ncd * (t - 1) + j - 1] = cd0;
            }
        }
        else{
            spd._f0_open = IloNumArray(env, d.np * d.nf);
            spd._cd0_open = IloNumArray(env, d.np * d.ncd);
            spd._f0_clos = IloNumArray(env, d.np * d.nf);
            spd._cd0_clos = IloNumArray(env, d.np * d.ncd);
            for (int k = 1; k <= d.nf; k++){
                spd._f0_open[k - 1] = f0;
                spd._f0_clos[k - 1] = 0;
            }
            for (int j = 1; j <= d.ncd; j++){
                spd._cd0_open[j - 1] = cd0;
                spd._cd0_clos[j - 1] = 0;
            }
            for (int t = 2; t <= d.np; t++){
                for (int k = 1; k <= d.nf; k++){
                    spd._f0_open[d.nf * (t - 1) + k - 1] = 0;
                    spd._f0_clos[d.nf * (t - 1) + k - 1] = 0;
                }
                for (int j = 1; j <= d.ncd; j++){
                    spd._cd0_open[d.ncd * (t - 1) + j - 1] = 0;
                    spd._cd0_clos[d.ncd * (t - 1) + j - 1] = 0;
                }
            }
        }        
        
        IloEnv env1 = mp.cplex.getEnv();
        if(d.mwcut == 1){
            
            mp.f0_m = IloNumArray(env1, d.np * d.nf);
            mp.cd0_m = IloNumArray(env1, d.np * d.ncd);
            for (int t = 1; t <= d.np; t++){
                for (int k = 1; k <= d.nf; k++) mp.f0_m[d.nf * (t - 1) + k - 1] = f0;
                for (int j = 1; j <= d.ncd; j++) mp.cd0_m[d.ncd * (t - 1) + j - 1] = cd0;
            }
        }
        
        if(d.mwpcut == 1){
            mp.f0_mw_func = IloNumArray(env1, d.np * d.nf);
            mp.cd0_mw_func = IloNumArray(env1, d.np * d.ncd);
            if (d.typemod == 1){
                for (int t = 1; t <= d.np; t++){
                    for (int k = 1; k <= d.nf; k++) mp.f0_mw_func[d.nf * (t - 1) + k - 1] = f0;
                    for (int j = 1; j <= d.ncd; j++) mp.cd0_mw_func[d.ncd * (t - 1) + j - 1] = cd0;
                }
            }
            else{
                mp.f0_mw_open = IloNumArray(env1, d.np * d.nf);
                mp.cd0_mw_open = IloNumArray(env1, d.np * d.ncd);
                mp.f0_mw_clos = IloNumArray(env1, d.np * d.nf);
                mp.cd0_mw_clos = IloNumArray(env1, d.np * d.ncd);
                for (int k = 1; k <= d.nf; k++){
                    mp.f0_mw_open[k - 1] = f0;
                    mp.f0_mw_clos[k - 1] = 0;
                }
                for (int j = 1; j <= d.ncd; j++){
                    mp.cd0_mw_open[j - 1] = cd0;
                    mp.cd0_mw_clos[j - 1] = 0;
                }
                for (int t = 2; t <= d.np; t++){
                    for (int k = 1; k <= d.nf; k++){
                        mp.f0_mw_open[d.nf * (t - 1) + k - 1] = 0;
                        mp.f0_mw_clos[d.nf * (t - 1) + k - 1] = 0;
                    }
                    for (int j = 1; j <= d.ncd; j++){
                        mp.cd0_mw_open[d.ncd * (t - 1) + j - 1] = 0;
                        mp.cd0_mw_clos[d.ncd * (t - 1) + j - 1] = 0;
                    }
                }
            }
        }
        
        crono.start();
        if (d.warm == 1){
            if(d.typemod == 1){
                IloConversion convf_open(mp.env, mp.f_open, ILOFLOAT);
                mp.mod.add(convf_open);
                IloConversion convf_clos(mp.env, mp.f_clos, ILOFLOAT);
                mp.mod.add(convf_clos);
                
                IloConversion convcd_open(mp.env, mp.cd_open, ILOFLOAT);
                mp.mod.add(convcd_open);
                IloConversion convcd_clos(mp.env, mp.cd_clos, ILOFLOAT);
                mp.mod.add(convcd_clos);

                IloConversion convf_func(mp.env, mp.f_func, ILOFLOAT);
                mp.mod.add(convf_func);
                IloConversion convcd_func(mp.env, mp.cd_func, ILOFLOAT);
                mp.mod.add(convcd_func);
            
                if (d.hlr == 1){
                    while ((stop == false)){
                        solve_papadakos(d, mp, spd);
                        add_n_cuts(d, mp, spd);
                        solve_mp(d, mp);
                        solve_spd(d, mp, spd);
                        add_n_cuts(d, mp, spd);
            
                        for (int t = 1; t <= d.np; t++){
                            for (int k = 1; k <= d.nf; k++) spd._f0_func[d.nf * (t - 1) + k - 1] = spd._f0_func[d.nf * (t - 1) + k - 1]*(1-LAMBDA) + mp._f_func[d.nf * (t - 1) + k - 1]*LAMBDA;
                            for (int j = 1; j <= d.ncd; j++) spd._cd0_func[d.ncd * (t - 1) + j - 1] = spd._cd0_func[d.ncd * (t - 1) + j - 1]*(1-LAMBDA) + mp._cd_func[d.ncd * (t - 1) + j - 1]*LAMBDA;
                        }
            
                        delta_lb = IloAbs(mp.lb - mp.old_lb);
                      
                        stop = (delta_lb < EPSILON) ? true : false;
                        mp.old_lb = mp.lb;
            
                    }
                }
                else{
                    while ((stop == false) && (ih < d.hh)){
                        solve_papadakos(d, mp, spd);
                        add_n_cuts(d, mp, spd);
                        solve_mp(d, mp);
                        solve_spd(d, mp, spd);
                        add_n_cuts(d, mp, spd);
                        
                        for (int t = 1; t <= d.np; t++){
                            for (int k = 1; k <= d.nf; k++) spd._f0_func[d.nf * (t - 1) + k - 1] = spd._f0_func[d.nf * (t - 1) + k - 1]*(1-LAMBDA) + mp._f_func[d.nf * (t - 1) + k - 1]*LAMBDA;
                            for (int j = 1; j <= d.ncd; j++) spd._cd0_func[d.ncd * (t - 1) + j - 1] = spd._cd0_func[d.ncd * (t - 1) + j - 1]*(1-LAMBDA) + mp._cd_func[d.ncd * (t - 1) + j - 1]*LAMBDA;
                        }
                        
                        delta_lb = IloAbs(mp.lb - mp.old_lb);
                        
                        ih++;
                        stop = (delta_lb < EPSILON) ? true : false;
                        mp.old_lb = mp.lb;
                        
                    }
                }    

                mp.mod.remove(convf_open);
                mp.mod.remove(convf_clos);
                
                mp.mod.remove(convcd_open);
                mp.mod.remove(convcd_clos);
                
                mp.mod.remove(convf_func);
                mp.mod.remove(convcd_func);
            }
            else{
                
                IloConversion convf_open(mp.env, mp.f_open, ILOFLOAT);
                mp.mod.add(convf_open);
                IloConversion convf_clos(mp.env, mp.f_clos, ILOFLOAT);
                mp.mod.add(convf_clos);
            
                IloConversion convcd_open(mp.env, mp.cd_open, ILOFLOAT);
                mp.mod.add(convcd_open);
                IloConversion convcd_clos(mp.env, mp.cd_clos, ILOFLOAT);
                mp.mod.add(convcd_clos);

                if (d.hlr == 1){
                    while ((stop == false)){
                        solve_papadakos(d, mp, spd);
                        add_n_cuts(d, mp, spd);
                        solve_mp(d, mp);
                        solve_spd(d, mp, spd);
                        add_n_cuts(d, mp, spd);
                    
                        for (int t = 1; t <= d.np; t++){
                            for (int k = 1; k <= d.nf; k++){
                                spd._f0_open[d.nf * (t - 1) + k - 1] = spd._f0_open[d.nf * (t - 1) + k - 1]*(1-LAMBDA) + mp._f_open[d.nf * (t - 1) + k - 1]*LAMBDA;
                                spd._f0_clos[d.nf * (t - 1) + k - 1] = spd._f0_clos[d.nf * (t - 1) + k - 1]*(1-LAMBDA) + mp._f_clos[d.nf * (t - 1) + k - 1]*LAMBDA;
                            }
                            for (int j = 1; j <= d.ncd; j++){
                                spd._cd0_open[d.ncd * (t - 1) + j - 1] = spd._cd0_open[d.ncd * (t - 1) + j - 1]*(1-LAMBDA) + mp._cd_open[d.ncd * (t - 1) + j - 1]*LAMBDA;
                                spd._cd0_clos[d.ncd * (t - 1) + j - 1] = spd._cd0_clos[d.ncd * (t - 1) + j - 1]*(1-LAMBDA) + mp._cd_clos[d.ncd * (t - 1) + j - 1]*LAMBDA;
                            }
                        }
                    
                        delta_lb = IloAbs(mp.lb - mp.old_lb);
                        stop = (delta_lb < EPSILON) ? true : false;
                        mp.old_lb = mp.lb;
                    }
                }
                else{
                    while ((stop == false) && (ih < d.hh)){
                        solve_papadakos(d, mp, spd);
                        add_n_cuts(d, mp, spd);
                        solve_mp(d, mp);
                        solve_spd(d, mp, spd);
                        add_n_cuts(d, mp, spd);
                        
                        for (int t = 1; t <= d.np; t++){
                            for (int k = 1; k <= d.nf; k++){
                                spd._f0_open[d.nf * (t - 1) + k - 1] = spd._f0_open[d.nf * (t - 1) + k - 1]*(1-LAMBDA) + mp._f_open[d.nf * (t - 1) + k - 1]*LAMBDA;
                                spd._f0_clos[d.nf * (t - 1) + k - 1] = spd._f0_clos[d.nf * (t - 1) + k - 1]*(1-LAMBDA) + mp._f_clos[d.nf * (t - 1) + k - 1]*LAMBDA;
                            }
                            for (int j = 1; j <= d.ncd; j++){
                                spd._cd0_open[d.ncd * (t - 1) + j - 1] = spd._cd0_open[d.ncd * (t - 1) + j - 1]*(1-LAMBDA) + mp._cd_open[d.ncd * (t - 1) + j - 1]*LAMBDA;
                                spd._cd0_clos[d.ncd * (t - 1) + j - 1] = spd._cd0_clos[d.ncd * (t - 1) + j - 1]*(1-LAMBDA) + mp._cd_clos[d.ncd * (t - 1) + j - 1]*LAMBDA;
                            }
                        }
                    
                        delta_lb = IloAbs(mp.lb - mp.old_lb);
                        
                        ih++;
                        stop = (delta_lb < EPSILON) ? true : false;
                        mp.old_lb = mp.lb;
                    }
                }    

                mp.mod.remove(convf_open);
                mp.mod.remove(convf_clos);
                    
                mp.mod.remove(convcd_open);
                mp.mod.remove(convcd_clos);
            }
        }
        
        mp.cplex.use(cutCallback(mp.cplex.getEnv(),d, mp, spd, spdi));
        mp.cplex.use(infoCallback(mp.cplex.getEnv(),d, mp, spd, spdi));
        if (d.ccall == 1){
            mp.cplex.use(cutCallback2(mp.cplex.getEnv(),d, mp, spd));
        }
        
            if (d.hcut == 1 && d.place == 1){
                IloNumVarArray startVar(env);
                IloNumArray startVal(env);
                for (int t = 1; t <= d.np; t++){
                    for (int j = 1; j <= d.ncd; j++){
                        startVar.add(mp.cd_open[d.ncd * (t - 1) + j - 1]);
                        startVal.add(h.cd_open_starG[t][j]);
                        startVar.add(mp.cd_clos[d.ncd * (t - 1) + j - 1]);
                        startVal.add(h.cd_clos_starG[t][j]);
                        if (d.typemod == 1){
                            startVar.add(mp.cd_func[d.ncd * (t - 1) + j - 1]);
                            startVal.add(h.cd_func_starG[t][j]);
                        }
                    }
                    for (int k = 1; k <= d.nf; k++) {
                        startVar.add(mp.f_open[d.nf * (t - 1) + k - 1]);
                        startVal.add(h.f_open_starG[t][k]);
                        startVar.add(mp.f_clos[d.nf * (t - 1) + k - 1]);
                        startVal.add(h.f_clos_starG[t][k]);
                        if (d.typemod == 1){
                            startVar.add(mp.f_func[d.nf * (t - 1) + k - 1]);
                            startVal.add(h.f_func_starG[t][k]);
                        }
                    }
                }
                mp.cplex.addMIPStart(startVar, startVal);
                startVal.end();
                startVar.end();
            }
        
        solve_mp(d, mp);
        double t_final = crono.getTime();
        mp.ub = mp.lb;
        
        printf ("FO: %.4f | Time: %.4f\n",mp.lb, t_final);
        //printf ("c%5d | %18.4f | %18.4f | %18.4f | %18.4f\n",ih, mp.lb, mp.ub, mp.lb - mp.ub, t_final);
        //print_solution(d, mp);
        //print_solution_new(d, mp);
        //print_solution_arq(d, h, mp, argv, t_final);
        
        //char nome2[250];
        char nome3[250];
        
        char stra[250] = "aux-DMA";
        //char strb[250] = "result-DMA";
            
        if (d.icut == 1){
            strcat (stra,"-I");
            //strcat (strb,"-I");
        }
        
        if (d.cfcut == 1){
            strcat (stra,"-CF");
            //strcat (strb,"-CF");
        }
        
        if (d.pcut == 1){
            strcat (stra,"-P");
            //strcat (strb,"-P");
        }
        
        if (d.mwcut == 1){
            strcat (stra,"-MW");
            //strcat (strb,"-MW");
        }
        
        if (d.iJVCcut == 1){
            strcat (stra,"-I-JVC");
            //strcat (strb,"-I-JVC");
        }
        
        if (d.iJVCcfcut == 1){
            strcat (stra,"-CF-JVC");
            //strcat (strb,"-CF-JVC");
        }
        
        if (d.mwpcut == 1){
            strcat (stra,"-MWp");
            //strcat (strb,"-MWp");
        }
        
        if(d.setPrio == 1){
            strcat (stra,"-SetPr");
            //strcat (strb,"-SetPr");
        }
        
        if(d.hcut == 1){
            strcat (stra,"-Heu");
            //strcat (strb,"-Heu");
            char  val[10];
            sprintf(val, "%i", d.typehcut); // inteiro para string
            strcat (stra, val);
        }
        
        //warm start?
        if (d.warm == 1){
            if (d.hlr == 1){
                strcat (stra,"-Wlr");
                //strcat (strb,"-Wlr");
            }
            else{
                char  val[10];
                sprintf(val, "%i", d.hh); // inteiro para string
                strcat (stra,"-W");
                //strcat (strb,"-W");
                strcat (stra, val);
                //strcat (strb, val);
            }
        }
        
        //cutCallback?
        if (d.ccall == 1){
            strcat (stra,"-CutC");
            //strcat (strb,"-CutC");
            char  val[10];
            if (d.ncuts > 0){
                sprintf(val, "%i", d.ncuts); // inteiro para string
                strcat (stra, val);
                //strcpy(nome3, stra);
                //strcpy(nome2, strb);
            }
            else strcat (stra,"All");
        }
        
        if (d.typemod == 1){
            strcat (stra,"-mod1.txt");
            //strcat (strb,"-mod1.txt");
            strcpy(nome3, stra);
            //strcpy(nome2, strb);
        }
        
        if (d.typemod == 2){
            strcat (stra,"-mod2.txt");
            //strcat (strb,"-mod2.txt");
            strcpy(nome3, stra);
            //strcpy(nome2, strb);
        }
        
        /*FILE *arq;
        arq = fopen(nome2, "aw+");
        fprintf (arq,"%s \n",argv[1]);
        fprintf(arq, "\n  H |          LB        |          SUP       |              Time \n");*/
        double gap = 0;
        if (d.hcut == 1) gap = 100*((h.fo_starG-mp.lb)/mp.lb);
        
        int nstat = d.sup.size();
        //apenas se não abrir nó nenhum  - instâncias pequenas
        if(nstat == 0){
            //fprintf (arq,"%3d | %18.4f | %18.4f | %18.4f\n",0, mp.lb, mp.ub, t_final);
            FILE *arq3;
            arq3 = fopen(nome3, "aw+");
            fprintf (arq3,"%8.4f | %8.4f | %16.4f | %8d | %16.4f | %16.4f | %12.4f |   %s\n",gap,d.time_heu,h.fo_starG,0, mp.lb, mp.ub, t_final, argv[1]);
            fclose(arq3);
        }
        else{
            FILE *arq3;
            arq3 = fopen(nome3, "aw+");
            fprintf (arq3,"%8.4f | %8.4f | %16.4f | %8d | %16.4f | %16.4f | %12.4f |   %s\n",gap, d.time_heu,h.fo_starG,nstat-1, d.lb[nstat-1], d.sup[nstat-1], d.time[nstat-1], argv[1]);
            fclose(arq3);
        }
        
        /*for (int k = 0; k < nstat; k++){
            fprintf (arq,"%3d | %18.4f | %18.4f | %18.4f\n",k, d.lb[k], d.sup[k], d.time[k]);
            if(k == nstat-1){
            FILE *arq3;
            arq3 = fopen(nome3, "aw+");
            fprintf (arq3,"%3d | %18.4f | %18.4f | %18.4f |   %s\n",k, d.lb[k], d.sup[k], d.time[k], argv[1]);
            fclose(arq3);
            }
        } 
        
        fprintf (arq,"\n \n");
        fclose(arq);// fecha o arquivo de armazenamento*/
        
    }
    catch (IloException& ex){
        cerr << "Error: " << ex << endl;
    } 
    return 0;
}

// ========================================================
// Função Help
// ========================================================
void help(){
  cout << endl << endl << "Help \n " << endl;
  exit(1);
}

// ========================================================
// Leitura dos dados
// ========================================================
void read_data(char name[], DAT &d){
    ifstream arq(name);
    if (!arq.is_open()){
        help();
    }
    strcpy(d.name, name);
    arq >> d.np;
    arq >> d.nc;
    arq >> d.ncd;
    arq >> d.nf;
    
    d.ccd_open = vector<vector<double > >(d.np + 1,vector<double>(d.ncd +1));  
    d.ccd_clos = vector<vector<double > >(d.np + 1,vector<double>(d.ncd +1));  
    d.ccd_func = vector<vector<double > >(d.np + 1,vector<double>(d.ncd +1));  
    d.cf_open = vector<vector<double > >(d.np + 1,vector<double>(d.nf +1));        
    d.cf_clos = vector<vector<double > >(d.np + 1,vector<double>(d.nf +1));        
    d.cf_func = vector<vector<double > >(d.np + 1,vector<double>(d.nf +1));        
    d.d = vector<vector<double > >(d.np + 1,vector<double>(d.nc +1));
    
    d.c1 = vector<vector<vector<double> > > (d.np + 1,vector<vector<double> >(d.nc +1,vector<double>(d.ncd +1)));
    d.c2 = vector<vector<vector<double> > > (d.np + 1,vector<vector<double> >(d.ncd +1,vector<double>(d.nf +1)));
    d.c = vector<vector<vector<vector<double> > > > (d.np + 1, vector<vector<vector<double> > >(d.nc + 1,vector<vector<double> >(d.ncd +1,vector<double>(d.nf +1))));
    
    for (int t = 1; t <= d.np; t++){
        for (int i = 1; i <= d.ncd; i++){
            arq >> d.ccd_open[t][i];
            d.ccd_clos[t][i] = d.ccd_open[t][i] * PERC_CLOS;
            d.ccd_func[t][i] = d.ccd_open[t][i] * PERC_FUNC;
        }
    }
    
    for (int t = 1; t <= d.np; t++){
        for (int i = 1; i <= d.nf; i++){
            arq >> d.cf_open[t][i];
            d.cf_clos[t][i] = d.cf_open[t][i] * PERC_CLOS;
            d.cf_func[t][i] = d.cf_open[t][i] * PERC_FUNC;
        }
    }
    
    for (int t = 1; t <= d.np; t++){
        for (int i = 1; i <= d.ncd; i++){
            for (int j = 1; j <= d.nf; j++) arq >> d.c2[t][i][j];
        }
    }
    
    for (int t = 1; t <= d.np; t++){
        for (int i = 1; i <= d.nc; i++){
            for (int j = 1; j <= d.ncd; j++) arq >> d.c1[t][i][j];
        }
    }
    
    for (int t = 1; t <= d.np; t++){
        for (int i = 1; i <= d.nc; i++) arq >> d.d[t][i];
    }
    
    
    for (int t = 1; t <= d.np; t++){
        for (int i = 1; i <= d.nc; i++){
            for (int j = 1; j <= d.ncd; j++){
                for (int k = 1; k <= d.nf; k++) d.c[t][i][j][k] = d.d[t][i] * (d.c1[t][i][j] + d.c2[t][j][k]);
            }
        }
    }
    arq.close();
}

// ========================================================
// Leitura dos dados closing facility
// ========================================================
void read_data_CF(DAT &d){
    
    d.best_arc = vector<vector<vector<int > > >(d.np + 1,vector<vector<int> >(d.nc + 1,vector<int>(7)));
    d.min_arc = vector<vector<vector<int > > >(d.np + 1,vector<vector<int> >(d.nc + 1,vector<int>(3)));
    d.dif_minCD = vector<vector<double> >(d.np + 1,vector<double>(d.nc + 1,0.0));
    d.dif_minF = vector<vector<double> >(d.np + 1,vector<double>(d.nc + 1,0.0));
    for (int t = 1; t <= d.np; t++){
        for (int i = 1; i <= d.nc; i++){
            d.best_arc[t][i][1] = d.c_minCD[t][i][1];
            d.best_arc[t][i][2] = d.c_minF[t][i][1];
            d.best_arc[t][i][3] = d.c_minCD[t][i][2];
            d.best_arc[t][i][4] = d.c_minF[t][i][2];
            
            int b = 2;
            if(d.c_minCD[t][i][1] == d.c_minCD[t][i][2]){
                do{
                    b++;
                } while (d.c_minCD[t][i][1] == d.c_minCD[t][i][b]);
                d.best_arc[t][i][5] = d.c_minCD[t][i][b];
                d.best_arc[t][i][6] = d.c_minF[t][i][b];
            }
            
            b = 2;
            if(d.c_minF[t][i][1] == d.c_minF[t][i][2]){
                do{
                    b++;
                } while (d.c_minF[t][i][1] == d.c_minF[t][i][b]);
                d.best_arc[t][i][5] = d.c_minCD[t][i][b];
                d.best_arc[t][i][6] = d.c_minF[t][i][b];
            }
        }
    }
}

// ========================================================
// Problema Mestre
// ========================================================
void create_model_mp(DAT &d, MP_CPX_DAT &mp, HEU_DAT &h){

    IloEnv env = mp.env;
    mp.mod = IloModel(env); 
    mp.cplex = IloCplex(mp.mod);
    
    mp.f_open = IloNumVarArray(env, d.np * d.nf, 0.0, 1.0, ILOINT);
    mp.f_clos = IloNumVarArray(env, d.np * d.nf, 0.0, 1.0, ILOINT);       
    mp.cd_open = IloNumVarArray(env, d.np * d.ncd, 0.0, 1.0, ILOINT);     
    mp.cd_clos = IloNumVarArray(env, d.np * d.ncd, 0.0, 1.0, ILOINT);
    
    mp._f_open = IloNumArray(env, d.np * d.nf);
    mp._f_clos = IloNumArray(env, d.np * d.nf);
    mp._cd_open = IloNumArray(env, d.np * d.ncd);
    mp._cd_clos = IloNumArray(env, d.np * d.ncd);
    
    if (d.typemod == 1){
        mp.f_func = IloNumVarArray(env, d.np * d.nf, 0.0, 1.0, ILOINT);
        mp.cd_func = IloNumVarArray(env, d.np * d.ncd, 0.0, 1.0, ILOINT);
        mp._f_func = IloNumArray(env, d.np * d.nf);
        mp._cd_func = IloNumArray(env, d.np * d.ncd);
    }
    
    if(d.typemod == 2 && (d.cfcut == 1 || d.iJVCcfcut == 1)){
        mp._f_func_CF = IloNumArray(env, d.np * d.nf);
        mp._cd_func_CF = IloNumArray(env, d.np * d.ncd);
    }
    
    mp.eta = IloNumVarArray(env,d.np * d.nc,0.0,+IloInfinity,ILOFLOAT);  
    mp.constraints = IloRangeArray(env);
    mp.cuts = IloRangeArray(env);
    
    if (d.hcut == 1){
        mp.v_aux = IloNumArray(env, d.np * d.nc);
        mp.u_aux = IloNumArray(env, d.np * d.nc * d.nf);
        mp.w_aux = IloNumArray(env, d.np * d.nc * d.ncd);
    }
    
    // ===============
    // Função objetivo do mestre
    // ===============
    if (d.typemod == 1){
        IloExpr xpfo(env);
        for (int t = 1; t <= d.np; t++){
            for (int k = 1; k <= d.nf; k++){
                xpfo += (d.cf_open[t][k] * mp.f_open[d.nf * (t - 1) + k - 1]);
                xpfo += (d.cf_clos[t][k] * mp.f_clos[d.nf * (t - 1) + k - 1]);
                xpfo += (d.cf_func[t][k] * mp.f_func[d.nf * (t - 1) + k - 1]);
            }
            for (int j = 1; j <= d.ncd; j++){
                xpfo += (d.ccd_open[t][j] * mp.cd_open[d.ncd * (t - 1) + j - 1]);
                xpfo += (d.ccd_clos[t][j] * mp.cd_clos[d.ncd * (t - 1) + j - 1]);
                xpfo += (d.ccd_func[t][j] * mp.cd_func[d.ncd * (t - 1) + j - 1]);
            }
            for (int i = 1; i <= d.nc; i++) xpfo += mp.eta[d.nc * (t - 1) + i - 1];
        }
        mp.fo = IloAdd(mp.mod, IloMinimize(env, xpfo));
        xpfo.end();
    }
    else{ //modelo 2
        IloExpr xpfo(env);
        for (int t = 1; t <= d.np; t++){
            for (int k = 1; k <= d.nf; k++){
                xpfo += (d.cf_open[t][k] * mp.f_open[d.nf * (t - 1) + k - 1]);
                xpfo += (d.cf_clos[t][k] * mp.f_clos[d.nf * (t - 1) + k - 1]);
                for (int r = 1; r <= t; r++) xpfo += (d.cf_func[t][k] * mp.f_open[d.nf * (r - 1) + k - 1]);
                for (int r = 2; r <= t; r++) xpfo -= (d.cf_func[t][k] * mp.f_clos[d.nf * (r - 1) + k - 1]);
            }
            for (int j = 1; j <= d.ncd; j++){
                xpfo += (d.ccd_open[t][j] * mp.cd_open[d.ncd * (t - 1) + j - 1]);
                xpfo += (d.ccd_clos[t][j] * mp.cd_clos[d.ncd * (t - 1) + j - 1]);
                for (int r = 1; r <= t; r++) xpfo += (d.ccd_func[t][j] * mp.cd_open[d.ncd * (r - 1) + j - 1]);
                for (int r = 2; r <= t; r++) xpfo -= (d.ccd_func[t][j] * mp.cd_clos[d.ncd * (r - 1) + j - 1]);
            }
            for (int i = 1; i <= d.nc; i++) xpfo += mp.eta[d.nc * (t - 1) + i - 1];
        }
        mp.fo = IloAdd(mp.mod, IloMinimize(env, xpfo));
        xpfo.end();
    }
  
    // ===============
    // Restrições do mestre
    // ===============
    if (d.typemod == 1){
        for (int t = 1; t <= d.np; t++){
            IloExpr r1(env);
            for (int k = 1; k <= d.nf; k++) r1 += mp.f_func[d.nf * (t - 1) + k - 1];
            mp.constraints.add(r1 >= 1);
            r1.end();
        }
        
        for (int t = 1; t <= d.np; t++){
            IloExpr r2(env);
            for (int j = 1; j <= d.ncd; j++) r2 += mp.cd_func[d.ncd * (t - 1) + j - 1];
            mp.constraints.add(r2 >= 1);
            r2.end();
        }
        
        for(int k = 1; k <= d.nf; k++){
            IloExpr r3(env);
            r3 += mp.f_func[k - 1];
            r3 -= mp.f_open[k - 1];
            mp.constraints.add(r3 == 0);
            r3.end();
        }
        
        for (int t = 2; t <= d.np; t++){
            for(int k = 1; k <= d.nf; k++){
                IloExpr r4(env);
                r4 += mp.f_func[d.nf * (t - 1) + k - 1];
                r4 -= mp.f_open[d.nf * (t - 1) + k - 1];
                r4 -= mp.f_func[d.nf * (t - 2) + k - 1];
                r4 += mp.f_clos[d.nf * (t - 1) + k - 1];
                mp.constraints.add(r4 == 0);
                r4.end();
            }
        }
        
        for(int j = 1; j <= d.ncd; j++){
            IloExpr r5(env);
            r5 += mp.cd_func[j - 1];
            r5 -= mp.cd_open[j - 1];
            mp.constraints.add(r5 == 0);
            r5.end();
        }
        
        for (int t = 2; t <= d.np; t++){
            for(int j = 1; j <= d.ncd; j++){
                IloExpr r6(env);
                r6 += mp.cd_func[d.ncd * (t - 1) + j - 1];
                r6 -= mp.cd_open[d.ncd * (t - 1) + j - 1];
                r6 -= mp.cd_func[d.ncd * (t - 2) + j - 1];
                r6 += mp.cd_clos[d.ncd * (t - 1) + j - 1];
                mp.constraints.add(r6 == 0);
                r6.end();
            }
        }
    }

    if (d.typemod == 2){
        
        for (int t = 1; t <= d.np; t++){
            IloExpr r7(env);
            for (int k = 1; k <= d.nf; k++){
                for (int r = 1; r <= t; r++){
                    r7 += mp.f_open[d.nf * (r - 1) + k - 1];
                }
            }
            for (int k = 1; k <= d.nf; k++){
                for (int r = 2; r <= t; r++){
                    r7 -= mp.f_clos[d.nf * (r - 1) + k - 1];
                }
            }
            mp.constraints.add(r7 >= 1);
            r7.end();
        }
        
        for (int t = 1; t <= d.np; t++){
            IloExpr r8(env);
            for (int j = 1; j <= d.ncd; j++){
                for (int r = 1; r <= t; r++){
                    r8 += mp.cd_open[d.ncd * (r - 1) + j - 1];
                }
            }
            for (int j = 1; j <= d.ncd; j++){
                for (int r = 2; r <= t; r++){
                    r8 -= mp.cd_clos[d.ncd * (r - 1) + j - 1];
                }
            }
            mp.constraints.add(r8 >= 1);
            r8.end();
        }

        
        for (int t = 1; t <= d.np; t++){
            for (int k = 1; k <= d.nf; k++){
                IloExpr r9(env);
                r9 = mp.f_clos[d.nf * (t - 1) + k - 1];
                for (int r = 1; r <= t-1; r++){
                    r9 -= mp.f_open[d.nf * (r - 1) + k - 1];
                    r9 += mp.f_clos[d.nf * (r - 1) + k - 1];
                }
                mp.constraints.add(r9 <= 0);
                r9.end();
            }
        }
        
        for (int t = 1; t <= d.np; t++){
            for (int j = 1; j <= d.ncd; j++){
                IloExpr r10(env);
                r10 = mp.cd_clos[d.ncd * (t - 1) + j - 1];
                for (int r = 1; r <= t-1; r++){
                    r10 -= mp.cd_open[d.ncd * (r - 1) + j - 1];
                    r10 += mp.cd_clos[d.ncd * (r - 1) + j - 1];
                }
                mp.constraints.add(r10 <= 0);
                r10.end();
            }
        }
    }
     
    if (d.hcut == 1){
        if (d.typehcut == 1 || d.typehcut == 3){
            
            IloExpr r12(env);
            for (int t = 1; t <= d.np; t++){
                for (int k = 1; k <= d.nf; k++){
                    r12 += (d.cf_open[t][k] * mp.f_open[d.nf * (t - 1) + k - 1]);
                    r12 += (d.cf_clos[t][k] * mp.f_clos[d.nf * (t - 1) + k - 1]);
                    for (int r = 1; r <= t; r++) r12 += (d.cf_func[t][k] * mp.f_open[d.nf * (r - 1) + k - 1]);
                    for (int r = 2; r <= t; r++) r12 -= (d.cf_func[t][k] * mp.f_clos[d.nf * (r - 1) + k - 1]);
                }
                for (int j = 1; j <= d.ncd; j++){
                    r12 += (d.ccd_open[t][j] * mp.cd_open[d.ncd * (t - 1) + j - 1]);
                    r12 += (d.ccd_clos[t][j] * mp.cd_clos[d.ncd * (t - 1) + j - 1]);
                    for (int r = 1; r <= t; r++) r12 += (d.ccd_func[t][j] * mp.cd_open[d.ncd * (r - 1) + j - 1]);
                    for (int r = 2; r <= t; r++) r12 -= (d.ccd_func[t][j] * mp.cd_clos[d.ncd * (r - 1) + j - 1]);
                }
                for (int i = 1; i <= d.nc; i++) r12 += mp.eta[d.nc * (t - 1) + i - 1];
            }
            mp.constraints.add(r12 <= h.fo_starG);
            r12.end();
        }
        
        if (d.typehcut == 2 || d.typehcut == 3){
            solve_spdiJVC_heu(d, mp, h);
            
            for (int t = 1; t <= d.np; t++){
                for (int i = 1; i <= d.nc; i++){
                    IloExpr r11(env);
                    r11 += mp.eta[d.nc * (t - 1) + i - 1];
                    r11 -= mp.v_aux[d.nc * (t - 1) + i - 1];
                    if (d.typemod == 1){
                        for (int k = 1; k <= d.nf; k++) r11 += mp.u_aux[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] * mp.f_func[d.nf * (t - 1) + k - 1];
                        for (int j = 1; j <= d.ncd; j++) r11 += mp.w_aux[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] * mp.cd_func[d.ncd * (t - 1) + j - 1];
                    }
                    else{
                        for (int k = 1; k <= d.nf; k++){
                            for (int r = 1; r <= t; r++) r11 += mp.u_aux[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] * mp.f_open[d.nf * (r - 1) + k - 1];
                            for (int r = 2; r <= t; r++) r11 -= mp.u_aux[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] * mp.f_clos[d.nf * (r - 1) + k - 1];
                        }
                        for (int j = 1; j <= d.ncd; j++){
                            for (int r = 1; r <= t; r++) r11 += mp.w_aux[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] * mp.cd_open[d.ncd * (r - 1) + j - 1];
                            for (int r = 2; r <= t; r++) r11 -= mp.w_aux[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] * mp.cd_clos[d.ncd * (r - 1) + j - 1];
                        }
                    }
                    mp.constraints.add(r11 >= 0);
                    r11.end();
                }
            }
        }
    }
    
    if (d.setPrio == 1){
        mp.varZ_open = IloNumArray(env, d.np * d.nf);
        mp.varZ_clos = IloNumArray(env, d.np * d.nf);
        mp.varY_open = IloNumArray(env, d.np * d.ncd);
        mp.varY_clos = IloNumArray(env, d.np * d.ncd);
        if (d.typemod == 1){
            mp.varZ_func = IloNumArray(env, d.np * d.nf);
            mp.varY_func = IloNumArray(env, d.np * d.ncd);
        }
    
        for (int t = 1; t <= d.np; t++){
            for (int k = 1; k <= d.nf; k++){
                mp.varZ_open[d.nf * (t - 1) + k - 1] = 2;
                mp.varZ_clos[d.nf * (t - 1) + k - 1] = 0;
                if (d.typemod == 1) mp.varZ_func[d.nf * (t - 1) + k - 1] = 2;
            }
            for (int j = 1; j <= d.ncd; j++){
                mp.varY_open[d.ncd * (t - 1) + j - 1] = 0;
                mp.varY_clos[d.ncd * (t - 1) + j - 1] = 0;
                if (d.typemod == 1) mp.varY_func[d.ncd * (t - 1) + j - 1] = 0;
            }
        }
        mp.cplex.setPriorities(mp.f_open, mp.varZ_open);
        mp.cplex.setPriorities(mp.f_clos, mp.varZ_clos);
        mp.cplex.setPriorities(mp.cd_open, mp.varY_open);
        mp.cplex.setPriorities(mp.cd_clos, mp.varY_clos);
        if (d.typemod == 1){
            mp.cplex.setPriorities(mp.f_func, mp.varZ_func);
            mp.cplex.setPriorities(mp.cd_func, mp.varY_func);
        }
    }

    //mp.cplex.setParam(IloCplex::Param::RandomSeed, 1);         //To define if are use the defaut seed or not
    mp.cplex.setParam(IloCplex::Threads,1);
    //mp.cplex.setWarning(env.getNullStream());
    //mp.cplex.setOut(env.getNullStream());
    mp.cplex.setParam(IloCplex::TiLim, 86400);
    mp.mod.add(mp.constraints);
    mp.mod.add(mp.cuts);
}

// ========================================================
// Criando o Subproblema dual
// ========================================================
void create_model_spd(DAT &d, MP_CPX_DAT &mp, SPD_CPX_DAT &spd){
  
    IloEnv env = spd.env;
    spd.mod = IloModel(env); 
    spd.cplex = IloCplex(spd.mod);

    spd.v = IloNumVarArray(env, d.np * d.nc, -IloInfinity, +IloInfinity, ILOFLOAT);
    spd.u = IloNumVarArray(env, d.np * d.nc * d.nf, 0.0, +IloInfinity, ILOFLOAT);
    spd.w = IloNumVarArray(env, d.np * d.nc * d.ncd, 0.0, +IloInfinity, ILOFLOAT);  
    
    spd._v = IloNumArray(env, d.np * d.nc);
    spd._u = IloNumArray(env, d.np * d.nc * d.nf);
    spd._w = IloNumArray(env, d.np * d.nc * d.ncd);
    
    spd._coef_f_func = IloNumArray(env, d.np * d.nc * d.nf);
    spd._coef_cd_func = IloNumArray(env, d.np * d.nc * d.ncd);
    
    spd.constraints = IloRangeArray(env);
    
    // ===============
    // Função objetivo do Subproblema dual
    // ===============
    if (d.typemod == 1){
        IloExpr xpfo(env);
        for (int t = 1; t <= d.np; t++){
            for (int i = 1; i <= d.nc; i++){
                xpfo += spd.v[d.nc * (t - 1) + i - 1];
                for (int k = 1; k <= d.nf; k++) xpfo -= mp._f_func[d.nf * (t - 1) + k - 1] * spd.u[d.nc * d.nf * (t - 1) + d.nf * (i - 1) + k - 1];
                for (int j = 1; j <= d.ncd; j++) xpfo -= mp._cd_func[d.ncd * (t - 1) + j - 1] * spd.w[d.nc * d.ncd * (t - 1) + d.ncd * (i - 1) + j - 1];
            }
        }
        spd.fo = IloAdd(spd.mod, IloMaximize(env, xpfo));
        xpfo.end();
    }
  
    if (d.typemod == 2){
        IloExpr xpfo(env);
        for (int t = 1; t <= d.np; t++){
            for (int i = 1; i <= d.nc; i++){
                xpfo += spd.v[d.nc * (t - 1) + i - 1];
                for (int k = 1; k <= d.nf; k++){
                    mp.valor = 0;
                    for (int r = 1; r <= t; r++) mp.valor += mp._f_open[d.nf * (r - 1) + k - 1];
                    for (int r = 2; r <= t; r++) mp.valor -= mp._f_clos[d.nf * (r - 1) + k - 1];
                    xpfo -= mp.valor * spd.u[d.nc * d.nf * (t - 1) + d.nf * (i - 1) + k - 1];
                }
                for (int j = 1; j <= d.ncd; j++){
                    mp.valor = 0;
                    for (int r = 1; r <= t; r++) mp.valor += mp._cd_open[d.ncd * (r - 1) + j - 1];
                    for (int r = 2; r <= t; r++) mp.valor -= mp._cd_clos[d.ncd * (r - 1) + j - 1];
                    xpfo -= mp.valor * spd.w[d.nc * d.ncd * (t - 1) + d.ncd * (i - 1) + j - 1];
                }
            }
        }
        spd.fo = IloAdd(spd.mod, IloMaximize(env, xpfo));
        xpfo.end();
    }
  
    // ===============
    // Restrições do Subproblema dual 
    // ===============
    for (int t = 1; t <= d.np; t++){
        for (int i = 1; i <= d.nc; i++){
            for (int j = 1; j <= d.ncd; j++){
                for (int k = 1; k <= d.nf; k++){
                    IloExpr r1(env);
                    r1 += spd.v[d.nc * (t - 1) + i - 1] - spd.u[d.nc * d.nf * (t - 1) + d.nf * (i - 1) + k - 1] - spd.w[d.nc * d.ncd * (t - 1) + d.ncd * (i - 1) + j - 1] - (d.c[t][i][j][k]);
                    spd.constraints.add(r1 <= 0);
                    r1.end();
                }
            }
        }
    }
    spd.cplex.setParam(IloCplex::Threads,1);
    spd.cplex.setWarning(env.getNullStream());
    spd.cplex.setOut(env.getNullStream());
    spd.mod.add(spd.constraints);
}

// ========================================================
// Criando o Subproblema dual por inspeção
// ========================================================
void create_model_spdi(DAT &d, SPDI_CPX_DAT &spdi){
    spdi.u = vector<vector<vector <double> > >(d.np + 1,vector<vector<double> >(d.nc +1,vector<double>(d.nf +1)));
    spdi.w = vector<vector<vector <double> > >(d.np + 1,vector<vector<double> >(d.nc +1,vector<double>(d.ncd +1)));
    spdi.v = vector<vector <double > >(d.np + 1,vector<double>(d.nc +1));
    
    spdi.dim = d.ncd;
    spdi.assigncost = new double*[spdi.dim+1];
    for (int i = 0; i < spdi.dim + 1; i++){
        spdi.assigncost[i] = new double[spdi.dim +1];
    }
    spdi.u_aux = new double[spdi.dim + 1];
    spdi.w_aux = new double[spdi.dim + 1];
    spdi.colsol = new int[spdi.dim + 1];
    spdi.rowsol = new int[spdi.dim + 1];
    
}

// ========================================================
// Resolvendo o Problema Mestre
// ========================================================
void solve_mp(DAT &d, MP_CPX_DAT &mp){
    mp.cplex.solve();
    mp.lb = (double) mp.cplex.getObjValue();
    
    mp.cplex.getValues(mp._f_clos, mp.f_clos);
    mp.cplex.getValues(mp._f_open, mp.f_open);
    
    mp.cplex.getValues(mp._cd_clos, mp.cd_clos); 
    mp.cplex.getValues(mp._cd_open, mp.cd_open);
    
    if (d.typemod == 1){
        mp.cplex.getValues(mp._f_func, mp.f_func);
        mp.cplex.getValues(mp._cd_func, mp.cd_func);
    }
}

// ========================================================
// Resolvendo o subproblema dual
// ========================================================
double solve_spd(DAT &d, MP_CPX_DAT &mp, SPD_CPX_DAT &spd){
    spd.of = 0.0;
    for (int t = 1; t <= d.np; t++){ 
        for (int i = 1; i <= d.nc; i++){
            if (d.typemod == 1){
                for (int k = 1; k <= d.nf; k++){
                    spd._coef_f_func[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] = -1.0 * mp._f_func[d.nf * (t - 1) + k - 1];
                }
                for (int j = 1; j <= d.ncd; j++){
                    spd._coef_cd_func[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] = -1.0 * mp._cd_func[d.ncd * (t - 1) + j - 1];
                }
            }
            else{
                for (int k = 1; k <= d.nf; k++){
                    mp.valor = 0;
                    for (int r = 1; r <= t; r++) mp.valor += mp._f_open[d.nf * (r - 1) + k - 1];
                    for (int r = 2; r <= t; r++) mp.valor -= mp._f_clos[d.nf * (r - 1) + k - 1];
                    spd._coef_f_func[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] = -1.0 * mp.valor;
                }
                for (int j = 1; j <= d.ncd; j++){
                    mp.valor = 0;
                    for (int r = 1; r <= t; r++) mp.valor += mp._cd_open[d.ncd * (r - 1) + j - 1];
                    for (int r = 2; r <= t; r++) mp.valor -= mp._cd_clos[d.ncd * (r - 1) + j - 1];
                    spd._coef_cd_func[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] = -1.0 * mp.valor;
                }
            }
        }
    }
    
    spd.fo.setLinearCoefs(spd.u, spd._coef_f_func);
    spd.fo.setLinearCoefs(spd.w, spd._coef_cd_func);

    spd.cplex.solve();
    spd.of = spd.cplex.getObjValue(); 
    spd.cplex.getValues(spd._v,spd.v);
    spd.cplex.getValues(spd._u,spd.u);
    spd.cplex.getValues(spd._w,spd.w);
}

// ========================================================
// Resolvendo o subproblema dual por inspeção
// ========================================================
double solve_spdi(DAT &d, MP_CPX_DAT &mp, SPDI_CPX_DAT &spdi){
    spdi._of = 0.0;
    create_vector_spdi(d, mp, spdi);
    for (int t = 1; t <= d.np; t++){
        for (int i = 1; i <= d.nc; i++){
            int a = valor_min(d, spdi, i, t);
            spdi.v[t][i] = d.c_minCusto[t][i][a];
            spdi._of += spdi.v[t][i];
            
            for (int j = 1; j <= d.ncd; j++){
                if (spdi.open_cd[t][j] == 1) spdi.w[t][i][j] = 0.0;
            }
            
            for (int k = 1; k <= d.nf; k++){
                if (spdi.open_fab[t][k] == 1) spdi.u[t][i][k] = 0.0;
            }
        
            double maior = -IloInfinity;
            for(int j = 1; j <= d.ncd; j++){
                if (spdi.open_cd[t][j] == 0){
                    for (int k = 1; k <= d.nf; k++){
                        if (spdi.open_fab[t][k] == 1){
                            if ( spdi.v[t][i] - d.c[t][i][j][k] > maior) maior = spdi.v[t][i] - d.c[t][i][j][k];
                        }
                    }
                    if (maior < zero) maior = 0.0;
                    spdi.w[t][i][j] = maior;
                    maior = -IloInfinity;
                }
            }
            
            maior = -IloInfinity;
            for(int k = 1; k <= d.nf; k++){
                if (spdi.open_fab[t][k] == 0){
                    for (int j = 1; j <= d.ncd; j++){
                        if (spdi.open_cd[t][j] == 1){
                            if ( spdi.v[t][i] - d.c[t][i][j][k] > maior) maior = spdi.v[t][i] - d.c[t][i][j][k];
                        }
                    }
                    if (maior < zero)maior = 0.0;
                    spdi.u[t][i][k] = maior;
                    maior = -IloInfinity;
                }
            }

            double val;
            for(int k = 1; k <= d.nf; k++){
                if (spdi.open_fab[t][k] == 0){
                    for (int j = 1; j <= d.ncd; j++){
                        if (spdi.open_cd[t][j] == 0){
                            if (spdi.u[t][i][k] + spdi.w[t][i][j] < spdi.v[t][i] - d.c[t][i][j][k]){
                                val = (spdi.v[t][i] - d.c[t][i][j][k]) - (spdi.u[t][i][k] + spdi.w[t][i][j]);
                                if (spdi.w[t][i][j] > spdi.u[t][i][k]) spdi.u[t][i][k] = spdi.u[t][i][k] + val;
                                else spdi.w[t][i][j] = spdi.w[t][i][j] + val;
                            }
                        }
                    }
                }
            }
        }
    }
    return spdi._of;
}

// ========================================================
// Função auxiliar para resolver o subproblema dual por inspeção
// ========================================================
void create_vector_spdi(DAT &d, MP_CPX_DAT &mp, SPDI_CPX_DAT &spdi){
    spdi.open_fab = vector<vector<int > >(d.np + 1,vector<int>(d.nf + 1));
    spdi.open_cd = vector<vector<int > >(d.np + 1,vector<int>(d.ncd + 1));
    for (int t = 1; t <= d.np; t++){
        if (d.typemod == 1){
            for (int k = 1; k <= d.nf; k++){
                if (mp._f_func[d.nf * (t - 1) + k - 1] > 0.9) spdi.open_fab[t][k] = 1;
                else spdi.open_fab[t][k] = 0;
            }
            
            for (int j = 1; j <= d.ncd; j++){
                if (mp._cd_func[d.ncd * (t - 1) + j - 1] > 0.9) spdi.open_cd[t][j] = 1;
                else spdi.open_cd[t][j] = 0;
            }
        }
        else{//modelo 2
            for (int k = 1; k <= d.nf; k++){
                double val = 0;
                for (int r = 1; r <= t; r++) val += mp._f_open[d.nf * (r - 1) + k - 1];
                for (int r = 2; r <= t; r++) val -= mp._f_clos[d.nf * (r - 1) + k - 1];
                if (val > 0.9) spdi.open_fab[t][k] = 1;
                else spdi.open_fab[t][k] = 0;
            }
                
            for (int j = 1; j <= d.ncd; j++){
                double val = 0;
                for (int r = 1; r <= t; r++) val += mp._cd_open[d.ncd * (r - 1) + j - 1];
                for (int r = 2; r <= t; r++) val -= mp._cd_clos[d.ncd * (r - 1) + j - 1];
                if (val > 0.9) spdi.open_cd[t][j] = 1;
                else spdi.open_cd[t][j] = 0;
            }
        }
    }
}

// ========================================================
// Aux SPDI
// ========================================================
int valor_min(DAT &d, SPDI_CPX_DAT &spdi, int i, int t){
    int a = 0;
    do a++;
    while(spdi.open_fab[t][(d.c_minF[t][i][a])] < 0.90 || spdi.open_cd[t][(d.c_minCD[t][i][a])] < 0.90);
    return a;
}

// ========================================================
// Aux function to CF
// ========================================================
void aux_CF(DAT &d, SPDI_CPX_DAT &spdi, int i, int t){
    int a = valor_min(d, spdi, i, t);
    d.min_arc[t][i][1] = d.c_minCD[t][i][a];
    d.min_arc[t][i][2] = d.c_minF[t][i][a];
    double dif;
    
    if(d.min_arc[t][i][1] != d.best_arc[t][i][1]){
        dif = d.c[t][i][d.best_arc[t][i][1]][d.best_arc[t][i][2]] - d.c[t][i][d.min_arc[t][i][1]][d.min_arc[t][i][2]];
        if(dif > zero) d.dif_minCD[t][i] = dif;
        else d.dif_minCD[t][i] = 0;
    }
    else{
        if (d.best_arc[t][i][1] != d.best_arc[t][i][3]){
            dif = d.c[t][i][d.best_arc[t][i][3]][d.best_arc[t][i][4]] - d.c[t][i][d.min_arc[t][i][1]][d.min_arc[t][i][2]];
            if(dif > zero) d.dif_minCD[t][i] = dif;
            else d.dif_minCD[t][i] = 0;
        }
    else{
            dif = d.c[t][i][d.best_arc[t][i][5]][d.best_arc[t][i][6]] - d.c[t][i][d.min_arc[t][i][1]][d.min_arc[t][i][2]];
            if(dif > zero) d.dif_minCD[t][i] = dif;
            else d.dif_minCD[t][i] = 0;
        }
    }
                
    if(d.min_arc[t][i][2] != d.best_arc[t][i][2]){
        dif = d.c[t][i][d.best_arc[t][i][1]][d.best_arc[t][i][2]] - d.c[t][i][d.min_arc[t][i][1]][d.min_arc[t][i][2]];
        if(dif > zero) d.dif_minF[t][i] = dif;
        else d.dif_minF[t][i] = 0;
    }
    else{
        if (d.best_arc[t][i][2] != d.best_arc[t][i][4]){
            dif = d.c[t][i][d.best_arc[t][i][3]][d.best_arc[t][i][4]] - d.c[t][i][d.min_arc[t][i][1]][d.min_arc[t][i][2]];
            if(dif > zero) d.dif_minF[t][i] = dif;
            else d.dif_minF[t][i] = 0;
        }
        else{
            dif = d.c[t][i][d.best_arc[t][i][5]][d.best_arc[t][i][6]] - d.c[t][i][d.min_arc[t][i][1]][d.min_arc[t][i][2]];
            if(dif > zero) d.dif_minF[t][i] = dif;
            else d.dif_minF[t][i] = 0;
        }
    }
}


// ========================================================
// Resolvendo o subproblema dual por inspeção JVC
// ========================================================
double solve_spdiJVC(DAT &d, MP_CPX_DAT &mp, SPDI_CPX_DAT &spdi){
    spdi._of = 0.0;
    create_vector_spdi(d, mp, spdi);
    for (int t = 1; t <= d.np; t++){
        for (int i = 1; i <= d.nc; i++){
            int a = valor_min(d, spdi, i, t);
            spdi.v[t][i] = d.c_minCusto[t][i][a];
            spdi._of += spdi.v[t][i];
            
            for (int j = 1; j <= d.ncd; j++){
                if (spdi.open_cd[t][j] == 1) spdi.w[t][i][j] = 0.0;
            }
            
            for (int k = 1; k <= d.nf; k++){
                if (spdi.open_fab[t][k] == 1) spdi.u[t][i][k] = 0.0;
            }
        
            double maior = -IloInfinity;
            for(int j = 1; j <= d.ncd; j++){
                if (spdi.open_cd[t][j] == 0){
                    for (int k = 1; k <= d.nf; k++){
                        if (spdi.open_fab[t][k] == 1){
                            if ( spdi.v[t][i] - d.c[t][i][j][k] > maior) maior = spdi.v[t][i] - d.c[t][i][j][k];
                        }
                    }
                    if (maior < zero) maior = 0.0;
                    spdi.w[t][i][j] = maior;
                    maior = -IloInfinity;
                }
            }
            
            maior = -IloInfinity;
            for(int k = 1; k <= d.nf; k++){
                if (spdi.open_fab[t][k] == 0){
                    for (int j = 1; j <= d.ncd; j++){
                        if (spdi.open_cd[t][j] == 1){
                            if ( spdi.v[t][i] - d.c[t][i][j][k] > maior) maior = spdi.v[t][i] - d.c[t][i][j][k];
                        }
                    }
                    if (maior < zero) maior = 0.0;
                    spdi.u[t][i][k] = maior;
                    maior = -IloInfinity;
                }
            }
            
            double val;
            for(int k = 1; k <= d.ncd; k++){
                if(k <= d.nf){
                    for (int j = 1; j <= d.ncd; j++){
                        if (spdi.open_fab[t][k] == 0 && spdi.open_cd[t][j] == 0){
                            if (spdi.u[t][i][k] + spdi.w[t][i][j] < spdi.v[t][i] - d.c[t][i][j][k]){
                                val = (spdi.v[t][i] - d.c[t][i][j][k]) - (spdi.u[t][i][k] + spdi.w[t][i][j]);
                                spdi.assigncost[k-1][j-1] = - val;
                            }
                            else spdi.assigncost[k-1][j-1] = 0;
                        }
                        else spdi.assigncost[k-1][j-1] = 0;
                    }
                }
                else{
                    for (int j = 1; j <= d.ncd; j++) spdi.assigncost[k-1][j-1] = 0;
                }  
            }

            double tcost = lap(spdi.dim, spdi.assigncost, spdi.rowsol, spdi.colsol, spdi.u_aux, spdi.w_aux);
            
            for(int k = 1; k <= d.nf; k++){
                if (spdi.open_fab[t][k] == 0){
                    for (int j = 1; j <= d.ncd; j++){
                        if (spdi.open_cd[t][j] == 0){
                            if (spdi.u[t][i][k] + spdi.w[t][i][j] < spdi.v[t][i] - d.c[t][i][j][k]){
                                if (spdi.u_aux[k-1] < zero) spdi.u[t][i][k] = spdi.u[t][i][k] - spdi.u_aux[k-1];
                                if (spdi.w_aux[j-1] < zero) spdi.w[t][i][j] = spdi.w[t][i][j] - spdi.w_aux[j-1];
                            }
                        }
                    }
                }
            }
        }
    }
    return spdi._of;
}

// ========================================================
// Resolvendo o subproblema dual por inspeção MW
// ========================================================
void solve_spdiMW(DAT &d, MP_CPX_DAT &mp, SPDI_CPX_DAT &spdi){

    double e_f[d.nf+1], e_cd[d.ncd+1], e_f_abs[d.nf+1], e_cd_abs[d.ncd+1];
    double v_aux[d.nc+1];
    create_vector_spdi(d, mp, spdi);
    for (int t = 1; t <= d.np; t++){
        for (int j = 1; j <= d.ncd; j++){
            e_cd[j] = spdi.open_cd[t][j] - mp.cd0_m[d.ncd * (t - 1) + j - 1]; 
            if (e_cd[j] < 0) e_cd_abs[j] = -e_cd[j];
            else e_cd_abs[j] = e_cd[j];
        }
        
        for (int k = 1; k <= d.nf; k++){
            e_f[k] = spdi.open_fab[t][k] - mp.f0_m[d.nf * (t - 1) + k - 1];
            if (e_f[k] < 0) e_f_abs[k] = -e_f[k];
            else e_f_abs[k] = e_f[k];
        }
  
        double spof = 0;
        for (int i = 1; i <= d.nc; i++){  
            int a = valor_min(d, spdi, i, t);
            double menor = d.c_minCusto[t][i][a];
            spof = spof + menor;
            
            double uUpsilon = IloInfinity, wUpsilon = IloInfinity;
            
            for(int k = 1; k <= d.nf; k++){
                spdi.u[t][i][k] = 0;
                if (k != d.c_minF[t][i][a] && uUpsilon > d.c[t][i][d.c_minCD[t][i][a]][k] - menor) uUpsilon = d.c[t][i][d.c_minCD[t][i][a]][k] - menor;
            }
            
            for(int j = 1; j <= d.ncd; j++){
                spdi.w[t][i][j] = 0;
                if (j != d.c_minCD[t][i][a] && wUpsilon > d.c[t][i][j][d.c_minF[t][i][a]] - menor) wUpsilon = d.c[t][i][j][d.c_minF[t][i][a]] - menor;
            }
            
            double totalUpsilon = wUpsilon + uUpsilon; 
            
            double L_cd = IloInfinity, L_fab = IloInfinity;
            for(int k = 1; k <= d.nf; k++){
                if(k != d.c_minF[t][i][a]){
                    for(int j = 1; j <= d.ncd; j++){
                        if(spdi.open_cd[t][j] == 1){
                            if(d.c[t][i][j][k] < L_fab) L_fab = d.c[t][i][j][k];
                        }
                    }
                }
            }
            
            for(int k = 1; k <= d.nf; k++){
                if(spdi.open_fab[t][k] == 1){
                    for(int j = 1; j <= d.ncd; j++){
                        if(j != d.c_minCD[t][i][a]){
                            if(d.c[t][i][j][k] < L_cd) L_cd = d.c[t][i][j][k];
                        }
                    }
                }
            }
            
            double L_max = L_fab;
            if (L_fab > L_cd) L_max = L_cd;
            
            double kappa = 10;
            double delta = (L_max - menor)/kappa;
            double paramv = menor;
            double Fmax = 0;
            
            bool stop = false;
            do{
                spdi.u[t][i][d.c_minF[t][i][a]] = (uUpsilon/totalUpsilon)*(paramv - menor);
                spdi.w[t][i][d.c_minCD[t][i][a]] = (wUpsilon/totalUpsilon)*(paramv - menor);
                
                double maximo;
                for(int k = 1; k <= d.nf; k++){
                    if(spdi.open_fab[t][k] == 0){
                        maximo = 0;
                        for(int j = 1; j <= d.ncd; j++){
                            if(spdi.open_cd[t][j] == 1){
                                if (maximo < (paramv - d.c[t][i][j][k] - spdi.w[t][i][j])) maximo = paramv - d.c[t][i][j][k] - spdi.w[t][i][j];
                            }
                        }
                        spdi.u[t][i][k] = maximo;
                    }
                }
                
                for(int j = 1; j <= d.ncd; j++){
                    if(spdi.open_cd[t][j] == 0){
                        maximo = 0;
                        for(int k = 1; k <= d.nf; k++){
                            if(spdi.open_fab[t][k] == 1){
                                if (maximo < (paramv - d.c[t][i][j][k] - spdi.u[t][i][k])) maximo = paramv - d.c[t][i][j][k] - spdi.u[t][i][k];
                            }
                        }
                        spdi.w[t][i][j] = maximo;
                    }
                }
                
                for(int k = 1; k <= d.nf; k++){
                    if(spdi.open_fab[t][k] == 0){
                        for(int j = 1; j <= d.ncd; j++){
                            if(spdi.open_cd[t][j] == 0){
                                if (paramv - d.c[t][i][j][k] > 0){
                                    double gamma = (paramv - d.c[t][i][j][k]) - (spdi.u[t][i][k] + spdi.w[t][i][j]); 
                                    if (gamma > 0){
                                        if (e_cd_abs[j] > e_f_abs[k]) spdi.u[t][i][k] = spdi.u[t][i][k] + gamma;
                                        else spdi.w[t][i][j] = spdi.w[t][i][j] + gamma;
                                    }
                                }
                            }
                        }
                    }
                }
                
                double F_v = paramv - (menor + e_f[d.c_minF[t][i][a]]*spdi.u[t][i][d.c_minF[t][i][a]] + e_cd[d.c_minCD[t][i][a]]*spdi.w[t][i][d.c_minCD[t][i][a]]);
                
                for(int k = 1; k <= d.nf; k++){
                    if(spdi.open_fab[t][k] == 0) F_v = F_v + e_f[k]*spdi.u[t][i][k];
                }
                for(int j = 1; j <= d.ncd; j++){
                    if(spdi.open_cd[t][j] == 0) F_v = F_v + e_cd[j]*spdi.w[t][i][j];
                }
                
                if (paramv >= L_max || F_v <= Fmax) stop = true;
                else{
                Fmax = F_v;
                    paramv = paramv + delta;
                }

            }while (stop == false);
            spdi.v[t][i] = paramv;  
        }
    }
}

// ========================================================
// Resolve o subproblema de Papadakos
// ========================================================
double solve_papadakos(DAT &d, MP_CPX_DAT &mp, SPD_CPX_DAT &spd){
    spd.of = 0.0;
    for (int t = 1; t <= d.np; t++){ 
        for (int i = 1; i <= d.nc; i++){
            if (d.typemod == 1){
                for (int k = 1; k <= d.nf; k++) spd._coef_f_func[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] = -1.0 * spd._f0_func[d.nf * (t - 1) + k - 1];
                for (int j = 1; j <= d.ncd; j++) spd._coef_cd_func[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] = -1.0 * spd._cd0_func[d.ncd * (t - 1) + j - 1];
            }
            else{
                for (int k = 1; k <= d.nf; k++){
                    mp.valor = 0;
                    for (int r = 1; r <= t; r++) mp.valor += spd._f0_open[d.nf * (r - 1) + k - 1];
                    for (int r = 2; r <= t; r++) mp.valor -= spd._f0_clos[d.nf * (r - 1) + k - 1];
                    spd._coef_f_func[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] = -1.0 * mp.valor;
                }
                for (int j = 1; j <= d.ncd; j++){
                    mp.valor = 0;
                    for (int r = 1; r <= t; r++) mp.valor += spd._cd0_open[d.ncd * (r - 1) + j - 1];
                    for (int r = 2; r <= t; r++) mp.valor -= spd._cd0_clos[d.ncd * (r - 1) + j - 1];
                    spd._coef_cd_func[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] = -1.0 * mp.valor;
                }                
            }
        }
    }
    
    spd.fo.setLinearCoefs(spd.u, spd._coef_f_func);
    spd.fo.setLinearCoefs(spd.w, spd._coef_cd_func);

    spd.cplex.solve();
    spd.of = spd.cplex.getObjValue(); 

    spd.of = spd.cplex.getObjValue(); 
    spd.cplex.getValues(spd._v,spd.v);
    spd.cplex.getValues(spd._u,spd.u);
    spd.cplex.getValues(spd._w,spd.w);
}

// ========================================================
// Resolvendo o subproblema dual de MW - Lemon
// ========================================================
void solve_spdiMWPareto(DAT &d, MP_CPX_DAT &mp, SPDI_CPX_DAT &spdi){
    create_vector_spdi(d, mp, spdi);
    double demanda;
    for (int t = 1; t <= d.np; t++){ 
        for (int i = 1; i <= d.nc; i++){
            demanda = 1;
            if (d.typemod == 1){
                for (int j = 1; j <= d.ncd; j++) demanda += mp.cd0_mw_func[d.ncd * (t - 1) + j - 1]; 
                for (int k = 1; k <= d.nf; k++) demanda += mp.f0_mw_func[d.nf * (t - 1) + k - 1];
            }
            else{
                for (int k = 1; k <= d.nf; k++){
                    mp.valor = 0;
                    for (int r = 1; r <= t; r++) mp.valor += mp.f0_mw_open[d.nf * (r - 1) + k - 1];
                    for (int r = 2; r <= t; r++) mp.valor -= mp.f0_mw_clos[d.nf * (r - 1) + k - 1];
                    demanda += mp.valor;
                    mp.f0_mw_func[d.nf * (t - 1) + k - 1] = mp.valor;
                }
                for (int j = 1; j <= d.ncd; j++){
                    mp.valor = 0;
                    for (int r = 1; r <= t; r++) mp.valor += mp.cd0_mw_open[d.ncd * (r - 1) + j - 1];
                    for (int r = 2; r <= t; r++) mp.valor -= mp.cd0_mw_clos[d.ncd * (r - 1) + j - 1];
                    demanda += mp.valor;
                    mp.cd0_mw_func[d.ncd * (t - 1) + j - 1] = mp.valor;
                }           
            }
            min_cost_flow(d, mp, spdi, t, i, demanda);
        }
    }
}

void min_cost_flow(DAT &d, MP_CPX_DAT &mp, SPDI_CPX_DAT &spdi, int t, int i, double demanda){
    Graph g;
    Node origem, destino, f_node[d.nf+1], cd_node[d.ncd+1], f_nodeAux[d.nf+1], cd_nodeAux[d.ncd+1];
    origem = g.addNode();
    destino = g.addNode();
    
    Arc arcs_origem[d.nf+1], arcs_destino[d.ncd+1], arcs[d.nf*d.ncd+1], arcs_first[d.nf+1], arcs_second[d.ncd+1];
    ArcMap<Weight> weights(g);
    ArcMap<Capacity> capacities(g);
    
    for (int k = 1; k <= d.nf; k++){
        f_node[k-1] = g.addNode();
        arcs_origem[k-1] = g.addArc(origem, f_node[k-1]);
        weights[arcs_origem[k-1]] = 0;
        capacities[arcs_origem[k-1]] = INT_MAX;
        
        f_nodeAux[k-1] = g.addNode();
        arcs_first[k-1] = g.addArc(f_node[k-1], f_nodeAux[k-1]);
        weights[arcs_first[k-1]] = 0;
        if (d.typemod == 1){
            if (spdi.open_fab[t][k] > 0) capacities[arcs_first[k-1]] = mp.f0_mw_func[d.nf * (t - 1) + k - 1] + demanda-1;
            else capacities[arcs_first[k-1]] = mp.f0_mw_func[d.nf * (t - 1) + k - 1];
        }
        else{
            mp.valor = 0;
            for (int r = 1; r <= t; r++) mp.valor += mp.f0_mw_open[d.nf * (r - 1) + k - 1];
            for (int r = 2; r <= t; r++) mp.valor -= mp.f0_mw_clos[d.nf * (r - 1) + k - 1];
            if (spdi.open_fab[t][k] > 0) capacities[arcs_first[k-1]] = mp.valor + demanda-1;
            else capacities[arcs_first[k-1]] = mp.valor;
        }
    }
    
    for (int j = 1; j <= d.ncd; j++){
        cd_node[j-1] = g.addNode();
        cd_nodeAux[j-1] = g.addNode();
        
        arcs_second[j-1] = g.addArc(cd_node[j-1], cd_nodeAux[j-1]);
        weights[arcs_second[j-1]] = 0;
        if (d.typemod == 1){
            if (spdi.open_cd[t][j] > 0) capacities[arcs_second[j-1]] = mp.cd0_mw_func[d.ncd * (t - 1) + j - 1] + demanda-1;
            else capacities[arcs_second[j-1]] = mp.cd0_mw_func[d.ncd * (t - 1) + j - 1];
        }
        else{
            mp.valor = 0;
            for (int r = 1; r <= t; r++) mp.valor += mp.cd0_mw_open[d.ncd * (r - 1) + j - 1];
            for (int r = 2; r <= t; r++) mp.valor -= mp.cd0_mw_clos[d.ncd * (r - 1) + j - 1];
            if (spdi.open_cd[t][j] > 0) capacities[arcs_second[j-1]] = mp.valor + demanda-1;
            else capacities[arcs_second[j-1]] = mp.valor;
        }
        
        arcs_destino[j-1] = g.addArc(cd_nodeAux[j-1], destino);
        weights[arcs_destino[j-1]] = d.d[t][i] * d.c1[t][i][j];
        capacities[arcs_destino[j-1]] = INT_MAX;
    }
    
    for (int k = 1; k <= d.nf; k++){
        for (int j = 1; j <= d.ncd; j++){
            arcs[d.ncd * (k-1) + j-1] = g.addArc(f_nodeAux[k-1], cd_node[j-1]);
            weights[arcs[d.ncd * (k-1) + j-1]] = d.d[t][i] * d.c2[t][j][k];
            capacities[arcs[d.ncd * (k-1) + j-1]] = INT_MAX;
        }
    }  
    
    NS ns(g);
    ns.costMap(weights).upperMap(capacities).stSupply(origem, destino, demanda);

    ArcMap<Capacity> flows(g);
    NS::ProblemType status = ns.run();
    switch (status){
    case NS::INFEASIBLE:
        cerr << "insufficient flow" << endl;
        break;
    case NS::OPTIMAL:
        ns.flowMap(flows);
        spdi.v[t][i] = -ns.potential(origem);

        for (int k = 1; k <= d.nf; k++){
            spdi.u[t][i][k] = 0;
            if (0 + ns.potential(f_node[k-1]) - ns.potential(f_nodeAux[k-1]) < zero && ns.flow(arcs_first[k-1]) > (capacities[arcs_first[k-1]] - zero) ){
                spdi.u[t][i][k] = -(0 + ns.potential(f_node[k-1]) - ns.potential(f_nodeAux[k-1]));
            }
        }
                
        for (int j = 1; j <= d.ncd; j++){
            spdi.w[t][i][j] = 0;
            if (0 + ns.potential(cd_node[j-1]) - ns.potential(cd_nodeAux[j-1]) < zero && ns.flow(arcs_second[j-1]) > (capacities[arcs_second[j-1]] - zero) ){
                spdi.w[t][i][j] = -(0 + ns.potential(cd_node[j-1]) - ns.potential(cd_nodeAux[j-1]));
            }
        }
        
        for (int k = 1; k <= d.nf; k++){
            for (int j = 1; j <= d.ncd; j++){
                if (spdi.v[t][i] - d.c[t][i][j][k] - spdi.w[t][i][j] - spdi.u[t][i][k] > 0){
                    if (mp.cd0_mw_func[d.ncd * (t - 1) + j - 1] > mp.f0_mw_func[d.nf * (t - 1) + k - 1]) spdi.u[t][i][k] += spdi.v[t][i] - d.c[t][i][j][k] - spdi.w[t][i][j] - spdi.u[t][i][k];
                    else spdi.w[t][i][j] += spdi.v[t][i] - d.c[t][i][j][k] - spdi.w[t][i][j] - spdi.u[t][i][k];
                }
            }
        }
        
        break;
    case NS::UNBOUNDED:
        cerr << "infinite flow" << endl;
        break;
    default:
        break;
    }
}

//=========================================================
// Adiciona cortes gerados pelo subproblema dual via CPLEX
// ========================================================
void add_n_cuts(DAT &d, MP_CPX_DAT &mp, SPD_CPX_DAT &spd){
    for (int t = 1; t <= d.np; t++){ 
        for (int i = 1; i <= d.nc; i++){
            IloEnv env = mp.cplex.getEnv();
            IloExpr cut(env);
            cut += mp.eta[d.nc * (t - 1) + i - 1];
            cut -= spd._v[d.nc * (t - 1) + i - 1];
            if (d.typemod == 1){
                for (int k = 1; k <= d.nf; k++) cut += spd._u[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] * mp.f_func[d.nf * (t - 1) + k - 1];
                for (int j = 1; j <= d.ncd; j++) cut += spd._w[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] * mp.cd_func[d.ncd * (t - 1) + j - 1];
            }
            else{
                for (int k = 1; k <= d.nf; k++){
                    for (int r = 1; r <= t; r++) cut += spd._u[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] * mp.f_open[d.nf * (r - 1) + k - 1];
                    for (int r = 2; r <= t; r++) cut -= spd._u[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] * mp.f_clos[d.nf * (r - 1) + k - 1];
                }
                for (int j = 1; j <= d.ncd; j++){
                    for (int r = 1; r <= t; r++) cut += spd._w[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] * mp.cd_open[d.ncd * (r - 1) + j - 1];
                    for (int r = 2; r <= t; r++) cut -= spd._w[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] * mp.cd_clos[d.ncd * (r - 1) + j - 1];
                }
            }
            mp.cuts.add(cut >= 0);
            mp.mod.add(mp.cuts);
            cut.end();
        }
    }
}

// ========================================================
// Imprime a Solução final
// ========================================================
void print_solution(DAT &d, MP_CPX_DAT &mp){
    for (int t = 1; t <= d.np; t++){
            printf(" Openned Facility First Level:\n");
            for (int k = 1; k <= d.nf; k++){
                if (mp._f_open[d.nf * (t - 1) + k - 1] > 0.9) printf("  %4d ", k);
            }
            printf("\n");
        
            printf(" Openned Facility Second Level :\n");
            for (int j = 1; j <= d.ncd; j++){
                if (mp._cd_open[d.ncd * (t - 1) + j - 1] > 0.9) printf("  %4d ", j);
            }
            printf("\n");
    
            printf(" Closed Facility First Level:\n");
            for (int k = 1; k <= d.nf; k++){
                if (mp._f_clos[d.nf * (t - 1) + k - 1] > 0.9) printf("  %4d ", k);
            }
            printf("\n");
            
            printf(" Closed Facility Second Level :\n");
            for (int j = 1; j <= d.ncd; j++){
                if (mp._cd_clos[d.ncd * (t - 1) + j - 1] > 0.9) printf("  %4d ", j);
            }
            printf("\n");
            
            
            if (d.typemod == 1){
                printf(" Functioning Facility First Level:\n");
                for (int k = 1; k <= d.nf; k++){
                    if (mp._f_func[d.nf * (t - 1) + k - 1] > 0.9) printf("  %4d ", k);
                }
                printf("\n");
                
                printf(" Functioning Facility Second Level :\n");
                for (int j = 1; j <= d.ncd; j++){
                    if (mp._cd_func[d.ncd * (t - 1) + j - 1] > 0.9) printf("  %4d ", j);
                }
                printf("\n");
                printf("\n ===================================================== \n");
            }
            else{
                printf(" Functioning Facility First Level:\n");
                for (int k = 1; k <= d.nf; k++){
                    double val = 0;
                    for (int r = 1; r <= t; r++) val += mp._f_open[d.nf * (r - 1) + k - 1];
                    for (int r = 2; r <= t; r++) val -= mp._f_clos[d.nf * (r - 1) + k - 1];
                    if (val > 0.9) printf("  %4d ", k);
                }
                printf("\n");
            
                printf(" Functioning Facility Second Level :\n");
                for (int j = 1; j <= d.ncd; j++){
                    double val = 0;
                    for (int r = 1; r <= t; r++) val += mp._cd_open[d.ncd * (r - 1) + j - 1];
                    for (int r = 2; r <= t; r++) val -= mp._cd_clos[d.ncd * (r - 1) + j - 1];
                    if (val > 0.9) printf("  %4d ", j);
                }
                printf("\n");
                printf("\n ===================================================== \n");
            }
    }
    
}


// ========================================================
// Imprime a Solução final
// ========================================================
void print_solution_new(DAT &d, MP_CPX_DAT &mp){
    printf("\n================================================================= \n");
    printf("First Level:\n");
    printf("      ");
    for (int t = 1; t <= d.np; t++) printf("|    t=%d    ", t);
    printf("\n Fac  ");
    for (int t = 1; t <= d.np; t++) printf("|  O  F  C  ");
    printf("\n");
    
    for (int k = 1; k <= d.nf; k++){
        int verifica = 0;
        for (int t = 1; t <= d.np; t++){
            if (mp._f_open[d.nf * (t - 1) + k - 1] > 0.9){
                verifica = 1;
                t = d.np + 1;
            }
        }
        if (verifica == 1){
            if(k>99) printf(" %d  ", k); //100 a 999
            else{
                if(k<10) printf("   %d  ", k); //1 a 9
                else printf("  %d  ", k); // 10 a 99
            }
            for (int t = 1; t <= d.np; t++){
                //open
                if (mp._f_open[d.nf * (t - 1) + k - 1] > 0.9) printf("|  %d", 1);
                else printf("|   ");
                //functioning
                if (d.typemod == 1){
                    if (mp._f_func[d.nf * (t - 1) + k - 1] > 0.9) printf("  %d", 1);
                    else printf("   ");
                }
                else{
                    double val = 0;
                    for (int r = 1; r <= t; r++) val += mp._f_open[d.nf * (r - 1) + k - 1];
                    for (int r = 2; r <= t; r++) val -= mp._f_clos[d.nf * (r - 1) + k - 1];
                    if (val > 0.9) printf("  %d", 1);
                    else printf("   ");
                }
                //closing
                if (mp._f_clos[d.nf * (t - 1) + k - 1] > 0.9) printf("  %d  ", 1);
                else printf("     ");
            }
            printf("\n");
        }
    }
    
    printf("\n\n");
    printf("Second Level:\n");
    printf("      ");
    for (int t = 1; t <= d.np; t++) printf("|    t=%d    ", t);
    printf("\n Fac  ");
    for (int t = 1; t <= d.np; t++) printf("|  O  F  C  ");
    printf("\n");
    
    for (int j = 1; j <= d.ncd; j++){
        int verifica = 0;
        for (int t = 1; t <= d.np; t++){
            if (mp._cd_open[d.ncd * (t - 1) + j - 1] > 0.9){
                verifica = 1;
                t = d.np + 1;
            }
        }
        if (verifica == 1){
            if(j>99) printf(" %d  ", j); //100 a 999 
            else{
                if(j<10) printf("   %d  ", j); //1 a 9
                else printf("  %d  ", j); // 10 a 99
            }
            for (int t = 1; t <= d.np; t++){
                //open
                if (mp._cd_open[d.ncd * (t - 1) + j - 1] > 0.9) printf("|  %d", 1);
                else printf("|   ");
                //functioning
                if (d.typemod == 1){
                    if (mp._cd_func[d.ncd * (t - 1) + j - 1] > 0.9) printf("  %d", 1);
                    else printf("   ");
                }
                else{
                    double val = 0;
                    for (int r = 1; r <= t; r++) val += mp._cd_open[d.ncd * (r - 1) + j - 1];
                    for (int r = 2; r <= t; r++) val -= mp._cd_clos[d.ncd * (r - 1) + j - 1];
                    if (val > 0.9) printf("  %d", 1);
                    else printf("   ");
                }
                //closing
                if (mp._cd_clos[d.ncd * (t - 1) + j - 1] > 0.9) printf("  %d  ", 1);
                else printf("     ");
            }
            printf("\n");
        }
    }
    printf("================================================================= \n");
    printf("Valor da FO: %f\n", mp.lb);
    printf("================================================================= \n");
}

/*=======================================================================================================================*/
// Apenas Funções associadas às heurísticas 
/*=======================================================================================================================*/
void read_data_heu(DAT &d, HEU_DAT &h){
    h.s = vector<vector<vector<int > > >(d.np + 1,vector<vector<int> >(d.nc + 1,vector<int>(2))); // Solução para cada (t,i)
    h.cd_open = vector<vector<bool > >(d.np + 1,vector<bool>(d.ncd + 1, false));
    h.cd_clos = vector<vector<bool > >(d.np + 1,vector<bool>(d.ncd + 1, false));
    h.cd_func = vector<vector<bool > >(d.np + 1,vector<bool>(d.ncd + 1, false));
    h.f_open = vector<vector<bool > >(d.np + 1,vector<bool>(d.nf + 1, false));
    h.f_clos = vector<vector<bool > >(d.np + 1,vector<bool>(d.nf + 1, false));
    h.f_func = vector<vector<bool > >(d.np + 1,vector<bool>(d.nf + 1, false));
    
    //melhor solução
    h.s_star = vector<vector<vector<int > > >(d.np + 1,vector<vector<int> >(d.nc + 1,vector<int>(2))); // Solução para cada (t,i)
    h.cd_open_star = vector<vector<bool > >(d.np + 1,vector<bool>(d.ncd + 1, false));
    h.cd_clos_star = vector<vector<bool > >(d.np + 1,vector<bool>(d.ncd + 1, false));
    h.cd_func_star = vector<vector<bool > >(d.np + 1,vector<bool>(d.ncd + 1, false));
    h.f_open_star = vector<vector<bool > >(d.np + 1,vector<bool>(d.nf + 1, false));
    h.f_clos_star = vector<vector<bool > >(d.np + 1,vector<bool>(d.nf + 1, false));
    h.f_func_star = vector<vector<bool > >(d.np + 1,vector<bool>(d.nf + 1, false));
    
    //melhor solução global
    h.s_starG = vector<vector<vector<int > > >(d.np + 1,vector<vector<int> >(d.nc + 1,vector<int>(2))); // Solução para cada (t,i)
    h.cd_open_starG = vector<vector<bool > >(d.np + 1,vector<bool>(d.ncd + 1, false));
    h.cd_clos_starG = vector<vector<bool > >(d.np + 1,vector<bool>(d.ncd + 1, false));
    h.cd_func_starG = vector<vector<bool > >(d.np + 1,vector<bool>(d.ncd + 1, false));
    h.f_open_starG = vector<vector<bool > >(d.np + 1,vector<bool>(d.nf + 1, false));
    h.f_clos_starG = vector<vector<bool > >(d.np + 1,vector<bool>(d.nf + 1, false));
    h.f_func_starG = vector<vector<bool > >(d.np + 1,vector<bool>(d.nf + 1, false));
    
    //cópias
    h.s_copia = vector<vector<vector<int > > >(d.np + 1,vector<vector<int> >(d.nc + 1,vector<int>(2))); // Solução para cada (t,i)
    h.cd_open_copia = vector<vector<bool > >(d.np + 1,vector<bool>(d.ncd + 1, false));
    h.cd_clos_copia = vector<vector<bool > >(d.np + 1,vector<bool>(d.ncd + 1, false));
    h.cd_func_copia = vector<vector<bool > >(d.np + 1,vector<bool>(d.ncd + 1, false));
    h.f_open_copia = vector<vector<bool > >(d.np + 1,vector<bool>(d.nf + 1, false));
    h.f_clos_copia = vector<vector<bool > >(d.np + 1,vector<bool>(d.nf + 1, false));
    h.f_func_copia = vector<vector<bool > >(d.np + 1,vector<bool>(d.nf + 1, false));
    
    //JVC
    h.dim = d.ncd;
    h.assigncost = new double*[h.dim+1];
    for (int i = 0; i < h.dim + 1; i++){
        h.assigncost[i] = new double[h.dim +1];
    }
    h.u_aux = new double[h.dim + 1];
    h.w_aux = new double[h.dim + 1];
    h.colsol = new int[h.dim + 1];
    h.rowsol = new int[h.dim + 1];
}

// ========================================================
//Função que ordena todos os períodos e clientes pelo custo mínimo de transporte
// ========================================================
void quicksort_mincost(DAT &d){
    d.c_minCusto = vector<vector<vector<double > > >(d.np + 1,vector<vector<double> >(d.nc + 1,vector<double>(d.ncd*d.nf + 1)));
    d.c_minF = vector<vector<vector<int > > >(d.np + 1,vector<vector<int> >(d.nc + 1,vector<int>(d.ncd*d.nf + 1)));
    d.c_minCD = vector<vector<vector<int > > >(d.np + 1,vector<vector<int> >(d.nc + 1,vector<int>(d.ncd*d.nf + 1)));
    
    d.trans_min = 0;
    for (int t = 1; t <= d.np; t++){
        for (int i = 1; i <= d.nc; i++){
            
            int ct = 1;
            for (int j = 1; j <= d.ncd; j++){
                for (int k = 1; k <= d.nf; k++){
                    d.c_minCusto[t][i][ct] = d.c[t][i][j][k];
                    d.c_minCD[t][i][ct] = j;
                    d.c_minF[t][i][ct] = k;
                    ct++;
                }
            }
            
            int inicio_global = 1;
            int fim_global = d.ncd*d.nf;
            quicksort(d, inicio_global, fim_global, i, t);
            d.trans_min += d.c_minCusto[t][i][1];
            
        }
    }
}

// ========================================================
//Função de ordenação adaptado
// ========================================================
void quicksort(DAT &d, int inicio_global, int fim_global, int i, int t){
    int ini, fim, meio, aux_pos;
    double aux_custo, pivo;
            
    ini = inicio_global;
    fim = fim_global;
            
    meio = (int) ((ini + fim) / 2);
    pivo = d.c_minCusto[t][i][meio];
            
    do{
        while (d.c_minCusto[t][i][ini] < pivo) ini++;
        while (d.c_minCusto[t][i][fim] > pivo) fim--;
                
        if(ini <= fim){
            aux_custo = d.c_minCusto[t][i][ini];
            d.c_minCusto[t][i][ini] = d.c_minCusto[t][i][fim];
            d.c_minCusto[t][i][fim] = aux_custo;
                
            aux_pos = d.c_minCD[t][i][ini];
            d.c_minCD[t][i][ini] = d.c_minCD[t][i][fim];
            d.c_minCD[t][i][fim] = aux_pos;
                    
            aux_pos = d.c_minF[t][i][ini];
            d.c_minF[t][i][ini] = d.c_minF[t][i][fim];
            d.c_minF[t][i][fim] = aux_pos;
            
            ini++;
            fim--;
        }
    }while(fim > ini);
    if(inicio_global < fim) quicksort(d, inicio_global, fim, i, t);
    if(ini < fim_global) quicksort(d, ini, fim_global, i, t);
}

// ========================================================
//Função que define h.s 
// ========================================================
void define_solucao(DAT &d, HEU_DAT &h){
    for (int t = 1; t <= d.np; t++){
        for (int i = 1; i <= d.nc; i++){
            int cont = 1;
            do
                if (h.cd_func[t][d.c_minCD[t][i][cont]] == false || h.f_func[t][d.c_minF[t][i][cont]] == false) cont++;
                else{ 
                    h.s[t][i][1] = cont;
                    cont = 0;
                }
            while (cont != 0);
        }
    }
}

// ========================================================
//Função que atualiza s_star 
// ========================================================
void update_s_star(DAT &d, HEU_DAT &h){
    for (int t = 1; t <= d.np; t++){
        for (int i = 1; i <= d.nc; i++) h.s_star[t][i][1] = h.s[t][i][1];
        for (int k = 1; k <= d.nf; k++){
            h.f_open_star[t][k] = h.f_open[t][k];
            h.f_clos_star[t][k] = h.f_clos[t][k];
            h.f_func_star[t][k] = h.f_func[t][k];
        }
        for (int j = 1; j <= d.ncd; j++){
            h.cd_open_star[t][j] = h.cd_open[t][j];
            h.cd_clos_star[t][j] = h.cd_clos[t][j];
            h.cd_func_star[t][j] = h.cd_func[t][j];
        }
    }
    h.fo_star = h.fo;
    h.trans_s_star = h.trans_s;
    h.instC_s_star = h.instC_s;
}

// ========================================================
//Função que calcula o valor da FO global
// ========================================================
double calcula_fo_global(DAT &d, HEU_DAT &h){
    double val = 0;
    h.trans_s = 0;
    h.instC_s = 0;
    for (int t = 1; t <= d.np; t++){
        for (int i = 1; i <= d.nc; i++){
            val += d.c_minCusto[t][i][h.s[t][i][1]];
            h.trans_s += d.c_minCusto[t][i][h.s[t][i][1]];
        }
        for (int k = 1; k <= d.nf; k++){
            if (h.f_open[t][k] == true){
                val += d.cf_open[t][k];
                h.instC_s += d.cf_open[t][k];
            }
            if (h.f_clos[t][k] == true){
                val += d.cf_clos[t][k];
                h.instC_s += d.cf_clos[t][k];
            }
            if (h.f_func[t][k] == true){
                val += d.cf_func[t][k];
                h.instC_s += d.cf_func[t][k];
            }
        }
        for (int j = 1; j <= d.ncd; j++){
            if (h.cd_open[t][j] == true){
                val += d.ccd_open[t][j];
                h.instC_s += d.ccd_open[t][j];
            }
            if (h.cd_clos[t][j] == true){
                val += d.ccd_clos[t][j];
                h.instC_s += d.ccd_clos[t][j];
            }
            if (h.cd_func[t][j] == true){
                val += d.ccd_func[t][j];
                h.instC_s += d.ccd_func[t][j];
            }
        }
    }
    return val;
}

// ========================================================
//Função que calcula o custo mínimo de instalação - apenas 1 facilidade por nível no primeiro período e manter nos próximos
// ========================================================
void F1(DAT &d, HEU_DAT &h){
    h.vc = vector<double>(d.nf*d.ncd + 1);
    h.a = vector<vector<int > >(d.nf*d.ncd + 1,vector<int>(2));
    double custo_total = 0;
    int contador = 1;
    h.minF1 = INT_MAX;
    h.maxF1 = INT_MIN;
    for (int k = 1; k <= d.nf; k++){
        for (int j = 1; j <= d.ncd; j++){
            custo_total += d.ccd_open[1][j] + d.cf_open[1][k];
            for (int t = 1; t <= d.np; t++){
                custo_total += d.ccd_func[t][j] + d.cf_func[t][k];
                for (int i = 1; i <= d.nc; i++) custo_total += d.c[t][i][j][k];
            }
            h.a[contador][0] = j;
            h.a[contador][1] = k;
            h.vc[contador] = custo_total;
            if(custo_total < h.minF1) h.minF1 = custo_total;
            if(custo_total > h.maxF1) h.maxF1 = custo_total;
            custo_total = 0;
            contador++;
        }
    }
    int inicio_global = 1;
    int fim_global = d.ncd*d.nf;
    quickF1(h, inicio_global, fim_global);
}

// ========================================================
//Função de ordenação para o GRASP
// ========================================================
void quickF1(HEU_DAT &h, int inicio_global, int fim_global){
    int ini, fim, meio, aux_pos;
    double aux_custo, pivo;
            
    ini = inicio_global;
    fim = fim_global;
            
    meio = (int) ((ini + fim) / 2);
    pivo = h.vc[meio];
    
    do{
        while (h.vc[ini] < pivo) ini++;
        while (h.vc[fim] > pivo) fim--;
                
        if(ini <= fim){
            aux_custo = h.vc[ini];
            h.vc[ini] = h.vc[fim];
            h.vc[fim] = aux_custo;
                
            aux_pos = h.a[ini][0];
            h.a[ini][0] = h.a[fim][0];
            h.a[fim][0] = aux_pos;
                    
            aux_pos = h.a[ini][1];
            h.a[ini][1] = h.a[fim][1];
            h.a[fim][1] = aux_pos;
            
            ini++;
            fim--;
        }
    }while(fim > ini);
    if(inicio_global < fim) quickF1(h, inicio_global, fim);
    if(ini < fim_global) quickF1(h, ini, fim_global);
}

// ========================================================
//Função que gera a solucao incial da heurística
// ========================================================
void solucao_inicial(DAT &d, HEU_DAT &h, int iter){
    
    if (iter > 1){
        if (h.fo_starG > h.fo_star) update_s_starG(d, h);
        zera_sol(d, h);
    }
    
    for (int t = 1; t <= d.np; t++){
        h.cd_func[t][h.a[1][0]] = true;
        h.f_func[t][h.a[1][1]] = true;
    }
    h.cd_open[1][h.a[1][0]] = true;
    h.f_open[1][h.a[1][1]] = true;
        
    define_solucao(d, h);
    h.fo = calcula_fo_global(d, h);
    update_s_star(d, h);

    for (int t = 1; t <= d.np; t++){
        h.cd_func[t][h.a[1][0]] = false;
        h.f_func[t][h.a[1][1]] = false;
    }
    h.cd_open[1][h.a[1][0]] = false;
    h.f_open[1][h.a[1][1]] = false;
    
    //srand(time(NULL));
    h.alpha = (rand()%21);                                      // Variando de 0% a 20%
    //h.alpha = (rand()%11) + 10;                               // Variando de 10% a 20%
    h.alpha = h.alpha/100;                                      //Parâmetro do metodo Grasp
    double fat_grasp = h.minF1 + h.alpha*(h.maxF1 - h.minF1);
    
    int contador = 1;
    while(h.vc[contador] <= fat_grasp) contador++;
    contador--;
    int aleatorio = (rand()%contador)+1;
    
    for (int t = 1; t <= d.np; t++){
        h.cd_func[t][h.a[aleatorio][0]] = true;
        h.f_func[t][h.a[aleatorio][1]] = true;
    }
    h.cd_open[1][h.a[aleatorio][0]] = true;
    h.f_open[1][h.a[aleatorio][1]] = true;
        
    define_solucao(d, h);
    h.fo = calcula_fo_global(d, h);
    //update_s_star(d, h);
}

// ========================================================
// Método GRASP
// ========================================================
void grasp(DAT &d, HEU_DAT &h){
    clock_t tInicio;
    tInicio = clock();
    h.fo_starG = INT_MAX;
    F1(d, h);
    for (int t = 1; t <= d.iter; t++){
        solucao_inicial(d, h, t);
        h.choose = 1;
        open_clos_grasp(d, h, 1);
        update_s(d, h);
        
        int a = close_test(d, h);
        if (a > 0){
            h.choose = 2;
            open_clos_grasp(d, h, 1);
            update_s(d, h);
            
            int statusH = typedemand(d, h);
            if (statusH > 0){
                try_changeS(d, h);
            }
        }
        if ((double( clock () - tInicio ) / CLOCKS_PER_SEC) > 60) t = d.iter + 1;
    }
    if (d.iter == 1) update_s_starG(d, h);
    
    create_vector(d, h);
}

// ========================================================
// Metodo de abertura exaustivo do GRASP
// ========================================================
void open_clos_grasp(DAT &d, HEU_DAT &h, int tt){
    bool sair = false;
    while (!sair){
        int aux = 0;
        do{
            aux = open_or_close_facility(d, h, 1, tt);
        }while(aux > 0);
        double fo_cd = h.fo_star;
        
        aux = 0;
        do{
            aux = open_or_close_facility(d, h, 2, tt);
        }while(aux > 0);
        
        if (h.choose == 1){
            aux = 0;
            do{
                aux = open_or_close_facility(d, h, 3, tt);
            }while(aux > 0);
        }
        if (fo_cd == h.fo_star) sair = true;
    }
}

// ========================================================
//ABRINDO FACILIDADES - INSERE O MAXIMO DE FACILIDADES POSSIVEL EM UM DADO NÍVEL (OU PAR)
// ========================================================
int open_or_close_facility(DAT &d, HEU_DAT &h, int level, int tt){
    
    h.fo_copia = h.fo;
    copia_sol(d, h);
    int avalia = 0;
    bool var, var2;
    if (h.choose == 1){
        var = false;
        var2 = true;
    }
    else{
        var = true;
        var2 = false;
    }
    
    if(level == 1){
        int conta_indice = 0;
        int *v_indice = new int[d.ncd+1];
        double *custo_indice = new double[d.ncd+1];
        for (int j = 1; j <= d.ncd; j++){
            if (h.cd_func[tt][j] == var){
                h.cd_open[tt][j] = var2;
                for (int t = tt; t <= d.np; t++) h.cd_func[t][j] = var2;
                define_solucao(d, h);
                h.fo = calcula_fo_global(d, h);
                avalia = verifica_sol(d, h, tt);
                if (h.fo_copia > h.fo){
                    conta_indice++;
                    v_indice[conta_indice] = j;
                    custo_indice[conta_indice] = h.fo;
                    if (h.fo_star > h.fo) update_s_star(d, h);
                }
                if (avalia == 0){
                    h.cd_open[tt][j] = var;
                    for (int t = tt; t <= d.np; t++) h.cd_func[t][j] = var;
                }
                else recover_sol(d, h);
            }
        }
        if(conta_indice > 0){
            quickCurrent(custo_indice, v_indice, 1, conta_indice);
            define_rand_solution(d,h,1, conta_indice, custo_indice, v_indice, tt);
            delete[] v_indice;
            delete[] custo_indice;
            return 1;
        }
        else{
            h.fo = h.fo_copia;
            recover_sol(d, h);
            delete[] v_indice;
            delete[] custo_indice;
            return 0;
        }
    }
    
    if(level == 2){
        int conta_indice = 0;
        int *v_indice = new int[d.nf+1];
        double *custo_indice = new double[d.nf+1];
        for (int k = 1; k <= d.nf; k++){
            if (h.f_func[tt][k] == var){
                h.f_open[tt][k] = var2;
                for (int t = tt; t <= d.np; t++) h.f_func[t][k] = var2;
                define_solucao(d, h);
                h.fo = calcula_fo_global(d, h);
                avalia = verifica_sol(d, h, tt);
                if (h.fo_copia > h.fo){
                    conta_indice++;
                    v_indice[conta_indice] = k;
                    custo_indice[conta_indice] = h.fo;
                    if (h.fo_star > h.fo) update_s_star(d, h);
                }
                if (avalia == 0){
                    h.f_open[tt][k] = var;
                    for (int t = tt; t <= d.np; t++) h.f_func[t][k] = var;
                }
                else recover_sol(d, h);
            }
        }
        
        if(conta_indice > 0){
            quickCurrent(custo_indice, v_indice, 1, conta_indice);
            define_rand_solution(d,h,2, conta_indice, custo_indice, v_indice, tt);
            delete[] v_indice;
            delete[] custo_indice;
            return 1;
        }
        else{
            h.fo = h.fo_copia;
            recover_sol(d, h);
            delete[] v_indice;
            delete[] custo_indice;
            return 0;
        }
    }

    if(level == 3){
        int conta_indice = 0;
        int *v_indice_fab = new int[d.ncd*d.nf+1];
        int *v_indice_cd = new int[d.ncd*d.nf+1];
        double *custo_indice = new double[d.ncd*d.nf+1];
        for (int k = 1; k <= d.nf; k++){
            if (h.f_func[tt][k] == false){
                h.f_open[tt][k] = true;
                for (int t = tt; t <= d.np; t++) h.f_func[t][k] = true;
                for (int j = 1; j <= d.ncd; j++){
                    if (h.cd_func[tt][j] == false){
                        h.cd_open[tt][j] = true;
                        for (int t = tt; t <= d.np; t++) h.cd_func[t][j] = true;
                        define_solucao(d, h);
                        h.fo = calcula_fo_global(d, h);
                        avalia = verifica_sol(d, h, tt);
                        if (h.fo_copia > h.fo){
                            conta_indice++;
                            v_indice_fab[conta_indice] = k;
                            v_indice_cd[conta_indice] = j;
                            custo_indice[conta_indice] = h.fo;
                            if (h.fo_star > h.fo) update_s_star(d, h);
                        }
                        if (avalia == 0){
                            h.cd_open[tt][j] = false;
                            for (int t = tt; t <= d.np; t++) h.cd_func[t][j] = false;
                        }
                        else{
                            recover_sol(d, h);
                            h.f_open[tt][k] = true;
                            for (int t = tt; t <= d.np; t++) h.f_func[t][k] = true;
                        }
                    }
                }
                h.f_open[tt][k] = false;
                for (int t = tt; t <= d.np; t++) h.f_func[t][k] = false;
            }
        }
        
        if(conta_indice > 0){
            quickCurrentFabCD(custo_indice, v_indice_fab, v_indice_cd, 1, conta_indice);
            define_rand_solutionFabCD(d,h,3, conta_indice, custo_indice, v_indice_fab, v_indice_cd, tt);
            delete[] v_indice_cd;
            delete[] v_indice_fab;
            delete[] custo_indice;
            return 1;
        }
        else{
            h.fo = h.fo_copia;
            recover_sol(d, h);
            delete[] v_indice_cd;
            delete[] v_indice_fab;
            delete[] custo_indice;
            return 0;
        }
    }
}

// ========================================================
//Cria uma cópia para backup
// ========================================================
void copia_sol(DAT &d, HEU_DAT &h){
    for (int t = 1; t <= d.np; t++){
        for (int j = 1; j <= d.ncd; j++){
            h.cd_func_copia[t][j] = h.cd_func[t][j];
            h.cd_open_copia[t][j] = h.cd_open[t][j];
            h.cd_clos_copia[t][j] = h.cd_clos[t][j];
        }
        
        for (int k = 1; k <= d.nf; k++){
            h.f_func_copia[t][k] = h.f_func[t][k];
            h.f_open_copia[t][k] = h.f_open[t][k];
            h.f_clos_copia[t][k] = h.f_clos[t][k];
        }
    }
}

// ========================================================
//Recupara a solução original
// ========================================================
void recover_sol(DAT &d, HEU_DAT &h){
    for (int t = 1; t <= d.np; t++){
        for (int j = 1; j <= d.ncd; j++){
            h.cd_func[t][j] = h.cd_func_copia[t][j];
            h.cd_open[t][j] = h.cd_open_copia[t][j];
            h.cd_clos[t][j] = h.cd_clos_copia[t][j];
        }
        
        for (int k = 1; k <= d.nf; k++){
            h.f_func[t][k] = h.f_func_copia[t][k];
            h.f_open[t][k] = h.f_open_copia[t][k];
            h.f_clos[t][k] = h.f_clos_copia[t][k];
        }
    }
}

// ========================================================
// Função para verificar se tem facilidade sem atender ninguem
// ========================================================
int verifica_sol(DAT &d, HEU_DAT &h, int tt){
    int entra = 0;
    int tem;
    for (int j = 1; j <= d.ncd; j++){
        tem = 0;
        if (h.cd_open[tt][j] == true){
            for (int i = 1; i <= d.nc; i++){
                if (d.c_minCD[tt][i][h.s[tt][i][1]] == j){
                    tem = 1;
                    i = d.nc + 1;
                }
            }
            if (tem == 0){
                h.cd_open[tt][j] = false;
                for (int t = tt; t <= d.np; t++) h.cd_func[t][j] = false;
                entra = 1;
            }
        }
    }
    for (int k = 1; k <= d.nf; k++){
        tem = 0;
        if (h.f_open[tt][k] == true){
            for (int i = 1; i <= d.nc; i++){
                if (d.c_minF[tt][i][h.s[tt][i][1]] == k){
                    tem = 1;
                    i = d.nc + 1;
                }
            }
            if (tem == 0){
                h.f_open[tt][k] = false;
                for (int t = tt; t <= d.np; t++) h.f_func[t][k] = false;
                entra = 1;
            }
        }
    }
    
    if (entra == 1){
        define_solucao(d, h);
        h.fo = calcula_fo_global(d, h);
        return 1;
    }
    else return 0;
}

// ========================================================
//Função de ordenação para o GRASP
// ========================================================
void quickCurrent(double *c, int *a, int inicio_global, int fim_global){
    int ini, fim, meio, aux_pos;
    double aux_custo, pivo;
            
    ini = inicio_global;
    fim = fim_global;
            
    meio = (int) ((ini + fim) / 2);
    pivo = c[meio];
    
    do{
        while (c[ini] < pivo) ini++;
        while (c[fim] > pivo) fim--;
                
        if(ini <= fim){
            aux_custo = c[ini];
            c[ini] = c[fim];
            c[fim] = aux_custo;
                
            aux_pos = a[ini];
            a[ini] = a[fim];
            a[fim] = aux_pos;
            
            ini++;
            fim--;
        }
    }while(fim > ini);
    if(inicio_global < fim) quickCurrent(c, a, inicio_global, fim);
    if(ini < fim_global) quickCurrent(c, a, ini, fim_global);
}

// ========================================================
//Escolhe a solução corrente de forma aleatoria
// ========================================================
void define_rand_solution(DAT &d, HEU_DAT &h, int level, int tamanho, double *c, int *a, int tt){
    //srand(time(NULL));
    h.alpha = (rand()%21);                                      // Variando de 0% a 20%
    //h.alpha = (rand()%11) + 10;                               // Variando de 10% a 20%
    h.alpha = h.alpha/100;                                      //Parâmetro do metodo Grasp
    double fat_grasp = c[1] + h.alpha*(c[tamanho] - c[1]);
    
    int contador = 1;
    while(c[contador] <= fat_grasp && contador <= tamanho) contador++;
    contador--;
    int aleatorio = (rand()%contador)+1;
    
    if (level == 1){
        if (h.cd_open[tt][a[aleatorio]] == false){
            h.cd_open[tt][a[aleatorio]] = true;
            for (int t = tt; t <= d.np; t++) h.cd_func[t][a[aleatorio]] = true;
            
            define_solucao(d, h);
            h.fo = calcula_fo_global(d, h);
            int avalia = verifica_sol(d, h, tt);
        }
        else{
            h.cd_open[tt][a[aleatorio]] = false;
            for (int t = tt; t <= d.np; t++) h.cd_func[t][a[aleatorio]] = false;
            
            define_solucao(d, h);
            h.fo = calcula_fo_global(d, h);
            int avalia = verifica_sol(d, h, tt);           
        }
        
    }
    
    if (level == 2){
        
        if (h.f_open[tt][a[aleatorio]] == false){
            h.f_open[tt][a[aleatorio]] = true;
            for (int t = tt; t <= d.np; t++) h.f_func[t][a[aleatorio]] = true;
            
            define_solucao(d, h);
            h.fo = calcula_fo_global(d, h);
            int avalia = verifica_sol(d, h, tt);
        }
        else{
            h.f_open[tt][a[aleatorio]] = false;
            for (int t = tt; t <= d.np; t++) h.f_func[t][a[aleatorio]] = false;
            
            define_solucao(d, h);
            h.fo = calcula_fo_global(d, h);
            int avalia = verifica_sol(d, h, tt);                     
        }
    }
}

// ========================================================
//Função de ordenação para o GRASP - open FAB and CD
// ========================================================
void quickCurrentFabCD(double *c, int *a, int *b, int inicio_global, int fim_global){
    int ini, fim, meio, aux_pos;
    double aux_custo, pivo;
            
    ini = inicio_global;
    fim = fim_global;
            
    meio = (int) ((ini + fim) / 2);
    pivo = c[meio];
    
    do{
        while (c[ini] < pivo) ini++;
        while (c[fim] > pivo) fim--;
                
        if(ini <= fim){
            aux_custo = c[ini];
            c[ini] = c[fim];
            c[fim] = aux_custo;
                
            aux_pos = a[ini];
            a[ini] = a[fim];
            a[fim] = aux_pos;
            
            aux_pos = b[ini];
            b[ini] = b[fim];
            b[fim] = aux_pos;
            
            ini++;
            fim--;
        }
    }while(fim > ini);
    if(inicio_global < fim) quickCurrentFabCD(c, a, b, inicio_global, fim);
    if(ini < fim_global) quickCurrentFabCD(c, a, b, ini, fim_global);
}

// ========================================================
//Escolhe a solução corrente de forma aleatoria - open FAB and CD
// ========================================================
void define_rand_solutionFabCD(DAT &d, HEU_DAT &h, int level, int tamanho, double *c, int *a, int *b, int tt){
    //srand(time(NULL));
    h.alpha = (rand()%21);                                      // Variando de 0% a 20%
    //h.alpha = (rand()%11) + 10;                               // Variando de 10% a 20%
    h.alpha = h.alpha/100;                                      //Parâmetro do metodo Grasp
    double fat_grasp = c[1] + h.alpha*(c[tamanho] - c[1]);
    
    int contador = 1;
    while(c[contador] <= fat_grasp && contador <= tamanho) contador++;
    contador--;
    int aleatorio = (rand()%contador)+1;
    
        h.f_open[tt][a[aleatorio]] = true;
        for (int t = tt; t <= d.np; t++) h.f_func[t][a[aleatorio]] = true;
        h.cd_open[tt][b[aleatorio]] = true;
        for (int t = tt; t <= d.np; t++) h.cd_func[t][b[aleatorio]] = true;
                
        define_solucao(d, h);
        h.fo = calcula_fo_global(d, h);
        int avalia = verifica_sol(d, h, tt);
}

// ========================================================
// Imprime a Solução final da Heurística
// ========================================================
void print_solution_heu(DAT &d, HEU_DAT &h){
    printf("\n================================================================= \n");
    printf("First Level:\n");
    printf("      ");
    for (int t = 1; t <= d.np; t++) printf("|    t=%d    ", t);
    printf("\n Fac  ");
    for (int t = 1; t <= d.np; t++) printf("|  O  F  C  ");
    printf("\n");
    
    for (int k = 1; k <= d.nf; k++){
        int verifica = 0;
        for (int t = 1; t <= d.np; t++){
            if (h.f_open_starG[t][k] > 0.9){
                verifica = 1;
                t = d.np + 1;
            }
        }
        if (verifica == 1){
            if(k>99) printf(" %d  ", k); //100 a 999
            else{
                if(k<10) printf("   %d  ", k); //1 a 9
                else printf("  %d  ", k); // 10 a 99
            }
            for (int t = 1; t <= d.np; t++){
                //open
                if (h.f_open_starG[t][k] > 0.9) printf("|  %d", 1);
                else printf("|   ");
                //functioning
                if (d.typemod == 1){
                    if (h.f_func_starG[t][k] > 0.9) printf("  %d", 1);
                    else printf("   ");
                }
                else{
                    double val = 0;
                    for (int r = 1; r <= t; r++) val += h.f_open_starG[r][k];
                    for (int r = 2; r <= t; r++) val -= h.f_clos_starG[r][k];
                    if (val > 0.9) printf("  %d", 1);
                    else printf("   ");
                }
                //closing
                if (h.f_clos_starG[t][k] > 0.9) printf("  %d  ", 1);
                else printf("     ");
            }
            printf("\n");
        }
    }
    
    printf("\n\n");
    printf("Second Level:\n");
    printf("      ");
    for (int t = 1; t <= d.np; t++) printf("|    t=%d    ", t);
    printf("\n Fac  ");
    for (int t = 1; t <= d.np; t++) printf("|  O  F  C  ");
    printf("\n");
    
    for (int j = 1; j <= d.ncd; j++){
        int verifica = 0;
        for (int t = 1; t <= d.np; t++){
            if (h.cd_open_starG[t][j] > 0.9){
                verifica = 1;
                t = d.np + 1;
            }
        }
        if (verifica == 1){
            if(j>99) printf(" %d  ", j); //100 a 999 
            else{
                if(j<10) printf("   %d  ", j); //1 a 9
                else printf("  %d  ", j); // 10 a 99
            }
            for (int t = 1; t <= d.np; t++){
                //open
                if (h.cd_open_starG[t][j] > 0.9) printf("|  %d", 1);
                else printf("|   ");
                //functioning
                if (d.typemod == 1){
                    if (h.cd_func_starG[t][j] > 0.9) printf("  %d", 1);
                    else printf("   ");
                }
                else{
                    double val = 0;
                    for (int r = 1; r <= t; r++) val += h.cd_open_starG[r][j];
                    for (int r = 2; r <= t; r++) val -= h.cd_clos_starG[r][j];
                    if (val > 0.9) printf("  %d", 1);
                    else printf("   ");
                }
                //closing
                if (h.cd_clos_starG[t][j] > 0.9) printf("  %d  ", 1);
                else printf("     ");
            }
            printf("\n");
        }
    }
    printf("================================================================= \n");
    printf("Valor da FO: %f\n", h.fo_starG);
    printf("================================================================= \n");     
}

// ========================================================
// Resolvendo o subproblema dual por inspeção - Heuristic
// ========================================================
void solve_spdi_heu(DAT &d, MP_CPX_DAT &mp, HEU_DAT &h){
    for (int t = 1; t <= d.np; t++){
        for (int i = 1; i <= d.nc; i++){
            mp.v_aux[d.nc * (t - 1) + i - 1] = d.c_minCusto[t][i][h.s_starG[t][i][1]];
                       
            for (int j = 1; j <= d.ncd; j++){
                if (h.open_cd[t][j] == 1) mp.w_aux[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] = 0.0;
            }
            
            for (int k = 1; k <= d.nf; k++){
                if (h.open_fab[t][k] == 1) mp.u_aux[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] = 0.0;
            }
            
            double maior = -IloInfinity;
            for(int j = 1; j <= d.ncd; j++){
                if (h.open_cd[t][j] == 0){
                    for (int k = 1; k <= d.nf; k++){
                        if (h.open_fab[t][k] == 1){
                            if ( mp.v_aux[d.nc * (t - 1) + i - 1] - d.c[t][i][j][k] > maior) maior = mp.v_aux[d.nc * (t - 1) + i - 1] - d.c[t][i][j][k];
                        }
                    }
                    if (maior < zero) maior = 0.0;
                    mp.w_aux[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] = maior;
                    maior = -IloInfinity;
                }
            }
            
            maior = -IloInfinity;
            for(int k = 1; k <= d.nf; k++){
                if (h.open_fab[t][k] == 0){
                    for (int j = 1; j <= d.ncd; j++){
                        if (h.open_cd[t][j] == 1){
                            if ( mp.v_aux[d.nc * (t - 1) + i - 1] - d.c[t][i][j][k] > maior) maior = mp.v_aux[d.nc * (t - 1) + i - 1] - d.c[t][i][j][k];
                        }
                    }
                    if (maior < zero) maior = 0.0;
                    mp.u_aux[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] = maior;
                    maior = -IloInfinity;
                }
            }

            double val;
            for(int k = 1; k <= d.nf; k++){
                if (h.open_fab[t][k] == 0){
                    for (int j = 1; j <= d.ncd; j++){
                        if (h.open_cd[t][j] == 0){
                            if (mp.u_aux[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] + mp.w_aux[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] < mp.v_aux[d.nc * (t - 1) + i - 1] - d.c[t][i][j][k]){
                                val = (mp.v_aux[d.nc * (t - 1) + i - 1] - d.c[t][i][j][k]) - (mp.u_aux[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] + mp.w_aux[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1]);
                                if (mp.w_aux[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] > mp.u_aux[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1]) mp.u_aux[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] = mp.u_aux[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] + val;
                                else mp.w_aux[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] = mp.w_aux[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] + val;
                            }
                        }
                    }
                }
            }
        }
    }
}

// ========================================================
// Resolvendo o subproblema dual por inspeção JVC - Heuristic
// ========================================================
void solve_spdiJVC_heu(DAT &d, MP_CPX_DAT &mp, HEU_DAT &h){
    for (int t = 1; t <= d.np; t++){
        for (int i = 1; i <= d.nc; i++){
            mp.v_aux[d.nc * (t - 1) + i - 1] = d.c_minCusto[t][i][h.s_starG[t][i][1]];
            
            for (int j = 1; j <= d.ncd; j++){
                if (h.open_cd[t][j] == 1) mp.w_aux[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] = 0.0;
            }
            
            for (int k = 1; k <= d.nf; k++){
                if (h.open_fab[t][k] == 1) mp.u_aux[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] = 0.0;
            }
        
            double maior = -IloInfinity;
            for(int j = 1; j <= d.ncd; j++){
                if (h.open_cd[t][j] == 0){
                    for (int k = 1; k <= d.nf; k++){
                        if (h.open_fab[t][k] == 1){
                            if ( mp.v_aux[d.nc * (t - 1) + i - 1] - d.c[t][i][j][k] > maior) maior = mp.v_aux[d.nc * (t - 1) + i - 1] - d.c[t][i][j][k];
                        }
                    }
                    if (maior < zero) maior = 0.0;
                    mp.w_aux[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] = maior;
                    maior = -IloInfinity;
                }
            }
            
            maior = -IloInfinity;
            for(int k = 1; k <= d.nf; k++){
                if (h.open_fab[t][k] == 0){
                    for (int j = 1; j <= d.ncd; j++){
                        if (h.open_cd[t][j] == 1){
                            if ( mp.v_aux[d.nc * (t - 1) + i - 1] - d.c[t][i][j][k] > maior) maior = mp.v_aux[d.nc * (t - 1) + i - 1] - d.c[t][i][j][k];
                        }
                    }
                    if (maior < zero) maior = 0.0;
                    mp.u_aux[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] = maior;
                    maior = -IloInfinity;
                }
            }
            
            double val;
            for(int k = 1; k <= d.ncd; k++){
                if(k <= d.nf){
                    for (int j = 1; j <= d.ncd; j++){
                        if (h.open_fab[t][k] == 0 && h.open_cd[t][j] == 0){
                            if (mp.u_aux[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] + mp.w_aux[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] < mp.v_aux[d.nc * (t - 1) + i - 1] - d.c[t][i][j][k]){
                                val = (mp.v_aux[d.nc * (t - 1) + i - 1] - d.c[t][i][j][k]) - (mp.u_aux[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] + mp.w_aux[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1]);
                                h.assigncost[k-1][j-1] = - val;
                            }
                            else h.assigncost[k-1][j-1] = 0;
                        }
                        else h.assigncost[k-1][j-1] = 0;
                    }
                }
                else{
                    for (int j = 1; j <= d.ncd; j++) h.assigncost[k-1][j-1] = 0;
                }  
            }

            double tcost = lap(h.dim, h.assigncost, h.rowsol, h.colsol, h.u_aux, h.w_aux);
            
            for(int k = 1; k <= d.nf; k++){
                if (h.open_fab[t][k] == 0){
                    for (int j = 1; j <= d.ncd; j++){
                        if (h.open_cd[t][j] == 0){
                            if (mp.u_aux[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] + mp.w_aux[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] < mp.v_aux[d.nc * (t - 1) + i - 1] - d.c[t][i][j][k]){
                                if (h.u_aux[k-1] < zero) mp.u_aux[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] = mp.u_aux[d.nf * d.nc * (t - 1) + d.nf * (i - 1) + k - 1] - h.u_aux[k-1];
                                if (h.w_aux[j-1] < zero) mp.w_aux[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] = mp.w_aux[d.ncd * d.nc * (t - 1) + d.ncd * (i - 1) + j - 1] - h.w_aux[j-1];
                            }
                        }
                    }
                }
            }
        }
    }
}

// ========================================================
// Função auxilia na resolução das inspeções - Heuristic
// ========================================================
void create_vector(DAT &d, HEU_DAT &h){
    h.open_fab = vector<vector<int > >(d.np + 1,vector<int>(d.nf + 1));
    h.open_cd = vector<vector<int > >(d.np + 1,vector<int>(d.ncd + 1));
    for (int t = 1; t <= d.np; t++){
        if (d.typemod == 1){
            for (int k = 1; k <= d.nf; k++){
                if (h.f_func_starG[t][k] > 0.9) h.open_fab[t][k] = 1;
                else h.open_fab[t][k] = 0;
            }
            
            for (int j = 1; j <= d.ncd; j++){
                if (h.cd_func_starG[t][j] > 0.9) h.open_cd[t][j] = 1;
                else h.open_cd[t][j] = 0;
            }
        }
        else{
            for (int k = 1; k <= d.nf; k++){
                double val = 0;
                for (int r = 1; r <= t; r++) val += h.f_open_starG[r][k];
                for (int r = 2; r <= t; r++) val -= h.f_clos_starG[r][k];
                if (val > 0.9) h.open_fab[t][k] = 1;
                else h.open_fab[t][k] = 0;
            }
                
            for (int j = 1; j <= d.ncd; j++){
                double val = 0;
                for (int r = 1; r <= t; r++) val += h.cd_open_starG[r][j];
                for (int r = 2; r <= t; r++) val -= h.cd_clos_starG[r][j];
                if (val > 0.9) h.open_cd[t][j] = 1;
                else h.open_cd[t][j] = 0;
            }
        }
    }
}

// ========================================================
//Função que atualiza s com a s_star 
// ========================================================
void update_s(DAT &d, HEU_DAT &h){
    for (int t = 1; t <= d.np; t++){
        for (int i = 1; i <= d.nc; i++) h.s[t][i][1] = h.s_star[t][i][1];
        for (int k = 1; k <= d.nf; k++){
            h.f_open[t][k] = h.f_open_star[t][k];
            h.f_clos[t][k] = h.f_clos_star[t][k];
            h.f_func[t][k] = h.f_func_star[t][k];
        }
        for (int j = 1; j <= d.ncd; j++){
            h.cd_open[t][j] = h.cd_open_star[t][j];
            h.cd_clos[t][j] = h.cd_clos_star[t][j];
            h.cd_func[t][j] = h.cd_func_star[t][j];
        }
    }
    h.fo = h.fo;
    h.trans_s = h.trans_s;
    h.instC_s = h.instC_s;
}

// ========================================================
// Imprime a Solução corrente da Heurística
// ========================================================
void print_solution_heu_current(DAT &d, HEU_DAT &h){
    printf("\n================================================================= \n");
    printf("First Level:\n");
    printf("      ");
    for (int t = 1; t <= d.np; t++) printf("|    t=%d    ", t);
    printf("\n Fac  ");
    for (int t = 1; t <= d.np; t++) printf("|  O  F  C  ");
    printf("\n");
    
    for (int k = 1; k <= d.nf; k++){
        int verifica = 0;
        for (int t = 1; t <= d.np; t++){
            if (h.f_open[t][k] > 0.9){
                verifica = 1;
                t = d.np + 1;
            }
        }
        if (verifica == 1){
            if(k>99) printf(" %d  ", k); //100 a 999
            else{
                if(k<10) printf("   %d  ", k); //1 a 9
                else printf("  %d  ", k); // 10 a 99
            }
            for (int t = 1; t <= d.np; t++){
                //open
                if (h.f_open[t][k] > 0.9) printf("|  %d", 1);
                else printf("|   ");
                //functioning
                if (d.typemod == 1){
                    if (h.f_func[t][k] > 0.9) printf("  %d", 1);
                    else printf("   ");
                }
                else{
                    double val = 0;
                    for (int r = 1; r <= t; r++) val += h.f_open[r][k];
                    for (int r = 2; r <= t; r++) val -= h.f_clos[r][k];
                    if (val > 0.9) printf("  %d", 1);
                    else printf("   ");
                }
                //closing
                if (h.f_clos[t][k] > 0.9) printf("  %d  ", 1);
                else printf("     ");
            }
            printf("\n");
        }
    }
    
    printf("\n\n");
    printf("Second Level:\n");
    printf("      ");
    for (int t = 1; t <= d.np; t++) printf("|    t=%d    ", t);
    printf("\n Fac  ");
    for (int t = 1; t <= d.np; t++) printf("|  O  F  C  ");
    printf("\n");
    
    for (int j = 1; j <= d.ncd; j++){
        int verifica = 0;
        for (int t = 1; t <= d.np; t++){
            if (h.cd_open[t][j] > 0.9){
                verifica = 1;
                t = d.np + 1;
            }
        }
        if (verifica == 1){
            if(j>99) printf(" %d  ", j); //100 a 999 
            else{
                if(j<10) printf("   %d  ", j); //1 a 9
                else printf("  %d  ", j); // 10 a 99
            }
            for (int t = 1; t <= d.np; t++){
                //open
                if (h.cd_open[t][j] > 0.9) printf("|  %d", 1);
                else printf("|   ");
                //functioning
                if (d.typemod == 1){
                    if (h.cd_func[t][j] > 0.9) printf("  %d", 1);
                    else printf("   ");
                }
                else{
                    double val = 0;
                    for (int r = 1; r <= t; r++) val += h.cd_open[r][j];
                    for (int r = 2; r <= t; r++) val -= h.cd_clos[r][j];
                    if (val > 0.9) printf("  %d", 1);
                    else printf("   ");
                }
                //closing
                if (h.cd_clos[t][j] > 0.9) printf("  %d  ", 1);
                else printf("     ");
            }
            printf("\n");
        }
    }
    printf("================================================================= \n");
    printf("Valor da FO: %f\n", h.fo);
    printf("================================================================= \n");     
}

// ========================================================
// Caracteriza o tipo de demanda - aceita um percentual
// ========================================================
int typedemand(DAT &d, HEU_DAT &h){
    int statusH = 0;
    h.demand = vector<int>(d.np + 1);
    h.media_demand = vector<double>(d.np + 1);
    h.demand[1] = 0;
    h.media_demand[1] = 0;
    for (int i = 1; i <= d.nc; i++) h.media_demand[1]+= d.d[1][i];
    h.media_demand[1] = h.media_demand[1]/d.nc;
    for (int t = 2; t <= d.np; t++){
        h.media_demand[t] = 0;
        for (int i = 1; i <= d.nc; i++) h.media_demand[t]+= d.d[t][i];
        h.media_demand[t] = h.media_demand[t]/d.nc;
                
        if (h.media_demand[t]/h.media_demand[t-1] <= 1-percH){
            h.demand[t] = 2;
            statusH++;
        }
        else{
            if (h.media_demand[t]/h.media_demand[t-1] > 1+percH){
                h.demand[t] = 1;
                 statusH++;
            }
            else h.demand[t] = 0;
        }
    }
    return statusH;
}

// ========================================================
// Tentantiva de melhorar a solução em função do tipo de demanda
// ========================================================
void try_changeS(DAT &d, HEU_DAT &h){
    h.choose = 1;
    for (int t = 2; t <= d.np; t++){
        if (h.demand[t] != 0){
            if (h.demand[t] == 1){
                closed_back_demand(d, h, t);
            }
            else{
                closed_front_demand(d, h, t);
            }
        }
    }
}

// ========================================================
// Metodo de fechamento exaustivo de fechamento de facilidades para frente
// ========================================================
void closed_front_demand(DAT &d, HEU_DAT &h, int tt){
    bool sair = false;
    while (!sair){
        int aux = 0;
        do{
            aux = closed_front(d, h, 1, tt);
        }while(aux > 0);
        double fo_cd = h.fo_star;
        
        aux = 0;
        do{
            aux = closed_front(d, h, 2, tt);
        }while(aux > 0);
        
        if (fo_cd == h.fo_star) sair = true;
    }
}

// ========================================================
//Fechando facilidades - para frente
// ========================================================
int closed_front(DAT &d, HEU_DAT &h, int level, int tt){
    h.fo_copia = h.fo;
    copia_sol(d, h);
    int a = 0;
    if(level == 1){
        int conta_indice = 0;
        int *v_indice = new int[d.ncd+1];
        double *custo_indice = new double[d.ncd+1];
        for (int j = 1; j <= d.ncd; j++){
            if (h.cd_func[tt][j] == true){
                h.cd_clos[tt][j] = true;
                for (int t = tt; t <= d.np; t++) h.cd_func[t][j] = false;
                if (h.choose == 2){
                    a = 1;
                    if (h.cd_open[tt][j] == true){
                        h.cd_open[tt][j] = false;
                        a = 0;
                    }
                }
                define_solucao(d, h);
                h.fo = calcula_fo_global(d, h);
                if (h.fo_copia > h.fo){
                    conta_indice++;
                    v_indice[conta_indice] = j;
                    custo_indice[conta_indice] = h.fo;
                    if (h.fo_star > h.fo) update_s_star(d, h);
                }
                h.cd_clos[tt][j] = false;
                for (int t = tt; t <= d.np; t++) h.cd_func[t][j] = true;
                if (h.choose == 2 && a == 0) h.cd_open[tt][j] = true;
            }
        }
        
        if(conta_indice > 0){
            quickCurrent(custo_indice, v_indice, 1, conta_indice);
            define_front_clos(d,h,1, conta_indice, custo_indice, v_indice, tt);
            delete[] v_indice;
            delete[] custo_indice;
            return 1;
        }
        else{
            h.fo = h.fo_copia;
            recover_sol(d, h);
            delete[] v_indice;
            delete[] custo_indice;
            return 0;
        }
    }
    
    if(level == 2){
        int conta_indice = 0;
        int *v_indice = new int[d.nf+1];
        double *custo_indice = new double[d.nf+1];
        
        for (int k = 1; k <= d.nf; k++){
            if (h.f_func[tt][k] == true){
                h.f_clos[tt][k] = true;
                for (int t = tt; t <= d.np; t++) h.f_func[t][k] = false;
                if (h.choose == 2){
                    a = 1;
                    if (h.f_open[tt][k] == true){
                        h.f_open[tt][k] = false;
                        a = 0;
                    }
                }
                define_solucao(d, h);
                h.fo = calcula_fo_global(d, h);
                
                if (h.fo_copia > h.fo){
                    
                    conta_indice++;
                    v_indice[conta_indice] = k;
                    custo_indice[conta_indice] = h.fo;
                    if (h.fo_star > h.fo) update_s_star(d, h);
                }
                h.f_clos[tt][k] = false;
                for (int t = tt; t <= d.np; t++) h.f_func[t][k] = true;
                if (h.choose == 2 && a == 0) h.f_open[tt][k] = true;
            }
        }
        
        if(conta_indice > 0){
            quickCurrent(custo_indice, v_indice, 1, conta_indice);
            define_front_clos(d,h,2, conta_indice, custo_indice, v_indice, tt);
            delete[] v_indice;
            delete[] custo_indice;
            return 1;
        }
        else{
            h.fo = h.fo_copia;
            recover_sol(d, h);
            delete[] v_indice;
            delete[] custo_indice;
            return 0;
        }
    }
}

// ========================================================
//Escolhe a facilidade fechada de forma aleatoria
// ========================================================
void define_front_clos(DAT &d, HEU_DAT &h, int level, int tamanho, double *c, int *a, int tt){
    //srand(time(NULL));
    h.alpha = (rand()%21);                                      // Variando de 0% a 20%
    //h.alpha = (rand()%11) + 10;                               // Variando de 10% a 20%
    h.alpha = h.alpha/100;                                      //Parâmetro do metodo Grasp
    double fat_grasp = c[1] + h.alpha*(c[tamanho] - c[1]);
    
    int contador = 1;
    while(c[contador] <= fat_grasp && contador <= tamanho) contador++;
    contador--;
    int aleatorio = (rand()%contador)+1;
    
    if (level == 1){
        h.cd_clos[tt][a[aleatorio]] = true;
        for (int t = tt; t <= d.np; t++) h.cd_func[t][a[aleatorio]] = false;
        if (h.cd_open[tt][a[aleatorio]] == true && h.choose == 2) h.cd_open[tt][a[aleatorio]] = false;
        define_solucao(d, h);
        h.fo = calcula_fo_global(d, h);
    }
    
    if (level == 2){
        h.f_clos[tt][a[aleatorio]] = true;
        for (int t = tt; t <= d.np; t++) h.f_func[t][a[aleatorio]] = false;
        if (h.f_open[tt][a[aleatorio]] == true && h.choose == 2) h.f_open[tt][a[aleatorio]] = false;
        define_solucao(d, h);
        h.fo = calcula_fo_global(d, h);
    }    
}

// ========================================================
// Metodo de fechamento exaustivo de fechamento de facilidades para frente
// ========================================================
void close_end_to_start(DAT &d, HEU_DAT &h){
    for (int t = d.np; t <= 2; t--){
        bool sair = false;
        while (!sair){
            int aux = 0;
            do{
                aux = closed_front(d, h, 1, t);
            }while(aux > 0);
            double fo_cd = h.fo_star;
            
            aux = 0;
            do{
                aux = closed_front(d, h, 2, t);
            }while(aux > 0);
                        
            if (fo_cd == h.fo_star) sair = true;
        }
    }
}

// ========================================================
// Metodo de fechamento exaustivo de fechamento de facilidades para trás
// ========================================================
void closed_back_demand(DAT &d, HEU_DAT &h, int tt){
    bool sair = false;
    while (!sair){
        int aux = 0;
        do{
            aux = closed_back(d, h, 1, tt);
        }while(aux > 0);
        double fo_cd = h.fo_star;
        
        aux = 0;
        do{
            aux = closed_back(d, h, 2, tt);
        }while(aux > 0);
        if (fo_cd == h.fo_star) sair = true;
    }
}

// ========================================================
//Fechando facilidades - para trás
// ========================================================
int closed_back(DAT &d, HEU_DAT &h, int level, int tt){
    h.fo_copia = h.fo;
    copia_sol(d, h);
    int avalia1 = 0;
    int avalia2 = 0;
    
    if(level == 1){
        int conta_indice = 0;
        int *v_indice = new int[d.ncd+1];
        double *custo_indice = new double[d.ncd+1];
        
        for (int j = 1; j <= d.ncd; j++){
            if (h.cd_func[tt][j] == true && h.cd_open[tt-1][j] == true){
                h.cd_open[tt][j] = true;
                h.cd_open[tt-1][j] = false;
                h.cd_func[tt-1][j] = false;
                define_solucao(d, h);
                h.fo = calcula_fo_global(d, h);
                
                avalia1 = verifica_sol(d, h, tt);
                avalia2 = verifica_sol(d, h, tt-1);
                if (h.fo_copia > h.fo){
                    
                    conta_indice++;
                    v_indice[conta_indice] = j;
                    custo_indice[conta_indice] = h.fo;
                    if (h.fo_star > h.fo) update_s_star(d, h);
                }
                if (avalia1 == 0 && avalia2 == 0){
                    h.cd_open[tt][j] = false;
                    h.cd_open[tt-1][j] = true;
                    h.cd_func[tt-1][j] = true;
                }
                else recover_sol(d, h);
            }
        }
        
        if(conta_indice > 0){
            quickCurrent(custo_indice, v_indice, 1, conta_indice);
            define_back_clos(d,h,1, conta_indice, custo_indice, v_indice, tt);
            delete[] v_indice;
            delete[] custo_indice;
            return 1;
        }
        else{
            h.fo = h.fo_copia;
            recover_sol(d, h);
            delete[] v_indice;
            delete[] custo_indice;
            return 0;
        }
    }
    
    if(level == 2){
        int conta_indice = 0;
        int *v_indice = new int[d.nf+1];
        double *custo_indice = new double[d.nf+1];
        
        for (int k = 1; k <= d.nf; k++){
            if (h.f_func[tt][k] == true && h.f_open[tt-1][k] == true){
                h.f_open[tt][k] = true;
                h.f_open[tt-1][k] = false;
                h.f_func[tt-1][k] = false;
                for (int t = tt; t <= d.np; t++) h.f_func[t][k] = false;
                define_solucao(d, h);
                h.fo = calcula_fo_global(d, h);
                
                avalia1 = verifica_sol(d, h, tt);
                avalia2 = verifica_sol(d, h, tt-1);
                if (h.fo_copia > h.fo){
                    
                    conta_indice++;
                    v_indice[conta_indice] = k;
                    custo_indice[conta_indice] = h.fo;
                    if (h.fo_star > h.fo) update_s_star(d, h);
                }
                if (avalia1 == 0 && avalia2 == 0){
                    h.f_open[tt][k] = false;
                    h.f_open[tt-1][k] = true;
                    h.f_func[tt-1][k] = true;
                }
                else recover_sol(d, h);
            }
        }
        
        if(conta_indice > 0){
            quickCurrent(custo_indice, v_indice, 1, conta_indice);
            define_back_clos(d,h,2, conta_indice, custo_indice, v_indice, tt);
            delete[] v_indice;
            delete[] custo_indice;
            return 1;
        }
        else{
            h.fo = h.fo_copia;
            recover_sol(d, h);
            delete[] v_indice;
            delete[] custo_indice;
            return 0;
        }
    }
}

// ========================================================
//Escolhe a facilidade fechada de forma aleatoria - para trás
// ========================================================
void define_back_clos(DAT &d, HEU_DAT &h, int level, int tamanho, double *c, int *a, int tt){
    //srand(time(NULL));
    h.alpha = (rand()%21);                                      // Variando de 0% a 20%
    //h.alpha = (rand()%11) + 10;                               // Variando de 10% a 20%
    h.alpha = h.alpha/100;                                      //Parâmetro do metodo Grasp
    double fat_grasp = c[1] + h.alpha*(c[tamanho] - c[1]);
    
    int contador = 1;
    while(c[contador] <= fat_grasp && contador <= tamanho) contador++;
    contador--;
    int aleatorio = (rand()%contador)+1;
    
    if (level == 1){
            h.cd_open[tt][a[aleatorio]] = true;
            h.cd_open[tt-1][a[aleatorio]] = false;
            h.cd_func[tt-1][a[aleatorio]] = false;
            
            define_solucao(d, h);
            h.fo = calcula_fo_global(d, h);
            int avalia1 = verifica_sol(d, h, tt);
            int avalia2 = verifica_sol(d, h, tt-1);  
    }
    
    if (level == 2){
            h.f_open[tt][a[aleatorio]] = true;
            h.f_open[tt-1][a[aleatorio]] = false;
            h.f_func[tt-1][a[aleatorio]] = false;
            
            define_solucao(d, h);
            h.fo = calcula_fo_global(d, h);
            int avalia1 = verifica_sol(d, h, tt);
            int avalia2 = verifica_sol(d, h, tt-1);
    }
}

// ========================================================
// Salva a solução final em um arquivo
// ========================================================
void print_solution_arq(DAT &d, HEU_DAT &h, MP_CPX_DAT &mp, char *argv[], double t_final){
    char nome[250];
    char stra[250] = "sol-DMA";
    if (d.icut == 1) strcat (stra,"-I");
    if (d.cfcut == 1) strcat (stra,"-CF");
    if (d.pcut == 1) strcat (stra,"-P");
    if (d.mwcut == 1) strcat (stra,"-MW");
    if (d.iJVCcut == 1) strcat (stra,"-I-JVC");
    if (d.iJVCcfcut == 1) strcat (stra,"-CF-JVC");
    if (d.mwcut == 1) strcat (stra,"-MWp");
    if(d.setPrio == 1) strcat (stra,"-SetPr");
    if(d.hcut == 1){
        strcat (stra,"-Heu");
        char  val[10];
        sprintf(val, "%i", d.typehcut);
        strcat (stra, val);
    }
    if (d.warm == 1){
        if (d.hlr == 1) strcat (stra,"-Wlr");
        else{
            char  val[10];
            sprintf(val, "%i", d.hh);
            strcat (stra,"-W");
            strcat (stra, val);
        }
    }
    if (d.ccall == 1){
        strcat (stra,"-CutC");
        char  val[10];
        if (d.ncuts > 0){
            sprintf(val, "%i", d.ncuts);
            strcat (stra, val);
        }
        else strcat (stra,"All");
    }
    if (d.typemod == 1){
        strcat (stra,"-mod1.txt");
        strcpy(nome, stra);
    }
    if (d.typemod == 2){
        strcat (stra,"-mod2.txt");
        strcpy(nome, stra);
    }
        
    FILE *arq;
    arq = fopen(nome, "aw+");
    fprintf (arq,"================================================================= \n");
    fprintf (arq,"Instance: %s\n", argv[1]);
    fprintf (arq,"Objective Function: %f\n", mp.lb);
    fprintf (arq,"Time: %f\n", t_final);
    fprintf (arq,"UPPER Bound heu: %f\n", h.fo_starG);
    fprintf (arq,"Time heu: %f\n", d.time_heu);
    fprintf (arq,"Gap (%): %f\n", 100*(h.fo_starG-mp.lb)/mp.lb);
    
    fprintf (arq,"================================================================= \n");
    fprintf (arq,"First Level:\n");
    fprintf (arq,"      ");
    for (int t = 1; t <= d.np; t++) fprintf (arq,"|    t=%d    ", t);
    fprintf (arq,"\n Fac  ");
    for (int t = 1; t <= d.np; t++) fprintf (arq,"|  O  F  C  ");
    fprintf (arq,"\n");
    
    for (int k = 1; k <= d.nf; k++){
        int verifica = 0;
        for (int t = 1; t <= d.np; t++){
            if (mp._f_open[d.nf * (t - 1) + k - 1] > 0.9){
                verifica = 1;
                t = d.np + 1;
            }
        }
        if (verifica == 1){
            if(k>99) fprintf (arq," %d  ", k); //100 a 999
            else{
                if(k<10) fprintf (arq,"   %d  ", k); //1 a 9
                else fprintf (arq,"  %d  ", k); // 10 a 99
            }
            for (int t = 1; t <= d.np; t++){
                //open
                if (mp._f_open[d.nf * (t - 1) + k - 1] > 0.9) fprintf (arq,"|  %d", 1);
                else fprintf (arq,"|   ");
                //functioning
                if (d.typemod == 1){
                    if (mp._f_func[d.nf * (t - 1) + k - 1] > 0.9) fprintf (arq,"  %d", 1);
                    else fprintf (arq,"   ");
                }
                else{
                    double val = 0;
                    for (int r = 1; r <= t; r++) val += mp._f_open[d.nf * (r - 1) + k - 1];
                    for (int r = 2; r <= t; r++) val -= mp._f_clos[d.nf * (r - 1) + k - 1];
                    if (val > 0.9) fprintf (arq,"  %d", 1);
                    else fprintf (arq,"   ");
                }
                //closing
                if (mp._f_clos[d.nf * (t - 1) + k - 1] > 0.9) fprintf (arq,"  %d  ", 1);
                else fprintf (arq,"     ");
            }
            fprintf (arq,"\n");
        }
    }
    
    fprintf (arq,"\n");
    fprintf (arq,"Second Level:\n");
    fprintf (arq,"      ");
    for (int t = 1; t <= d.np; t++) fprintf (arq,"|    t=%d    ", t);
    fprintf (arq,"\n Fac  ");
    for (int t = 1; t <= d.np; t++) fprintf (arq,"|  O  F  C  ");
    fprintf (arq,"\n");
    
    for (int j = 1; j <= d.ncd; j++){
        int verifica = 0;
        for (int t = 1; t <= d.np; t++){
            if (mp._cd_open[d.ncd * (t - 1) + j - 1] > 0.9){
                verifica = 1;
                t = d.np + 1;
            }
        }
        if (verifica == 1){
            if(j>99) fprintf (arq," %d  ", j); //100 a 999 
            else{
                if(j<10) fprintf (arq,"   %d  ", j); //1 a 9
                else fprintf (arq,"  %d  ", j); // 10 a 99
            }
            for (int t = 1; t <= d.np; t++){
                //open
                if (mp._cd_open[d.ncd * (t - 1) + j - 1] > 0.9) fprintf (arq,"|  %d", 1);
                else fprintf (arq,"|   ");
                //functioning
                if (d.typemod == 1){
                    if (mp._cd_func[d.ncd * (t - 1) + j - 1] > 0.9) fprintf (arq,"  %d", 1);
                    else fprintf (arq,"   ");
                }
                else{
                    double val = 0;
                    for (int r = 1; r <= t; r++) val += mp._cd_open[d.ncd * (r - 1) + j - 1];
                    for (int r = 2; r <= t; r++) val -= mp._cd_clos[d.ncd * (r - 1) + j - 1];
                    if (val > 0.9) fprintf (arq,"  %d", 1);
                    else fprintf (arq,"   ");
                }
                //closing
                if (mp._cd_clos[d.ncd * (t - 1) + j - 1] > 0.9) fprintf (arq,"  %d  ", 1);
                else fprintf (arq,"     ");
            }
            fprintf (arq,"\n");
        }
    }
    fprintf (arq,"================================================================= \n");
    fprintf (arq,"\n\n\n");
    fclose(arq);
}

// ========================================================
// Zerar variáveis 
// ========================================================
void zera_sol(DAT &d, HEU_DAT &h){
    for (int t = 1; t <= d.np; t++){
        for (int k = 1; k <= d.nf; k++){
            h.f_open_star[t][k] = 0;
            h.f_clos_star[t][k] = 0;
            h.f_func_star[t][k] = 0;
            h.f_open[t][k] = 0;
            h.f_clos[t][k] = 0;
            h.f_func[t][k] = 0;
        }
        for (int j = 1; j <= d.ncd; j++){
            h.cd_open_star[t][j] = 0;
            h.cd_clos_star[t][j] = 0;
            h.cd_func_star[t][j] = 0;
            h.cd_open[t][j] = 0;
            h.cd_clos[t][j] = 0;
            h.cd_func[t][j] = 0;
        }
    }
    h.fo_star = 0;
    h.fo = 0;
}

// ========================================================
//Função que atualiza s com a s_starG
// ========================================================
void update_s_starG(DAT &d, HEU_DAT &h){
    for (int t = 1; t <= d.np; t++){
        for (int i = 1; i <= d.nc; i++) h.s_starG[t][i][1] = h.s_star[t][i][1];
        for (int k = 1; k <= d.nf; k++){
            h.f_open_starG[t][k] = h.f_open_star[t][k];
            h.f_clos_starG[t][k] = h.f_clos_star[t][k];
            h.f_func_starG[t][k] = h.f_func_star[t][k];
        }
        for (int j = 1; j <= d.ncd; j++){
            h.cd_open_starG[t][j] = h.cd_open_star[t][j];
            h.cd_clos_starG[t][j] = h.cd_clos_star[t][j];
            h.cd_func_starG[t][j] = h.cd_func_star[t][j];
        }
    }
    h.fo_starG = h.fo_star;
    h.trans_s_starG = h.trans_s_star;
    h.instC_s_starG = h.instC_s_star;
}

int close_test(DAT &d, HEU_DAT &h){
    int fab = 0, cd = 0;
    for (int j = 1; j <= d.ncd; j++){
        if(h.cd_func[1][j] > 0) cd++; 
    }
    for (int k = 1; k <= d.nf; k++){
        if(h.f_func[1][k] > 0) fab++; 
    }
    if (fab > 1 && cd > 1){
        return 1;
    }
    else{
        return 0;
    }
}
