// ==============================================
// Autor: Paganini Barcellos de Oliveira
// ==============================================
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <list>
#include <cmath>
using namespace std;
#include <ilcplex/ilocplex.h>
#define EPSILON 0.000001
#define PERC_CLOS 0.1
#define PERC_FUNC 0.3
ILOSTLBEGIN

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
    vector <vector<double> > cf_open;                         // Custo de instalação de Fabricas - openning - 100%
    vector <vector<double> > cf_clos;                         // Custo de instalação de Fabricas - closing - 30%
    vector <vector<double> > cf_func;                         // Custo de instalação de Fabricas - functioning - 10%
    vector <vector<double> > d;                               // Demanda de cada cliente
    vector <vector<vector<double> > > c1;                     // Custo de transporte do CD j para o Cliente i
    vector <vector<vector<double> > > c2;                     // Custo de transporte da Fabrica k para o CD j
    vector <vector<vector<vector<double> > > > c;             // Custo de transporte total - open - 100%
    char name[250];
    int typemod;
    
    vector <double> lb;
    vector <double> sup;
    vector <double> ub;
    vector <double> time;
} DAT;

typedef struct{
    IloEnv env;
    IloCplex cplex;
    IloModel mod;
    IloNumVarArray x;
    IloNumVarArray w;
    IloNumArray _w;
    IloNumVarArray f_open;
    IloNumVarArray f_clos;
    IloNumVarArray f_func;
    IloNumArray _f_open;
    IloNumArray _f_clos;
    IloNumArray _f_func;
    IloNumVarArray cd_open;
    IloNumVarArray cd_clos;
    IloNumVarArray cd_func;
    IloNumArray _cd_open;
    IloNumArray _cd_clos;
    IloNumArray _cd_func;
    IloRangeArray constraints;
    IloObjective fo;
    IloTimer *crono;
    IloNum of;
} CPX_DAT;

ILOMIPINFOCALLBACK2(infoCallback, DAT &, d, CPX_DAT &, mono){
    if (hasIncumbent() == IloTrue){
        d.sup.push_back(getIncumbentObjValue());
        d.lb.push_back(getBestObjValue());
        double tt = mono.crono->getTime(); 
        d.time.push_back(tt);
    }
}

// ==============================================
// Funções auxiliares
// ==============================================
void read_data(char name[],DAT &d);
void help();
void create_model (DAT &d, CPX_DAT &mono);
void solve_model (DAT &d, CPX_DAT &mono);
void print_solution_mono(DAT &d, CPX_DAT &mono);
void print_solution_mono_new(DAT &d, CPX_DAT &mono);
void print_solution_arq(DAT &d, CPX_DAT &mono, char *argv[], int typeBenders, double t_final);

// ==============================================
// Função Principal
// ==============================================
int main (int argc, char *argv[]){
    DAT d;
    int typeBenders = (argc > 2) ? atoi(argv[2]) : 0; // Codes:(0) Without Benders or (1) Annotation or (2) FULL
    d.typemod = (argc > 3) ? atoi(argv[3]) : 1;       // Model:(1) S model or (2) Sỹ model or (3) Sỹza model or (4) Sỹzb model
    read_data(argv[1],d);                                 // Leitura dos dados

    // ===============
    // Ambiente cplex
    // ===============
    CPX_DAT mono;
    IloTimer crono(mono.env);
    mono.crono = &crono;
    create_model(d, mono);
    crono.start();
    mono.cplex.use(infoCallback(mono.cplex.getEnv(),d, mono));
    
    if (typeBenders == 1){
        //versão annotations
        IloCplex::LongAnnotation decomp = mono.cplex.newLongAnnotation(IloCplex::BendersAnnotation,CPX_BENDERS_MASTERVALUE+1);
        for (IloInt t = 1; t <= d.np; t++){
            for (IloInt j = 1; j <= d.ncd; j++){
                mono.cplex.setAnnotation(decomp, mono.cd_open[d.ncd * (t - 1) + j - 1], CPX_BENDERS_MASTERVALUE);
                mono.cplex.setAnnotation(decomp, mono.cd_clos[d.ncd * (t - 1) + j - 1], CPX_BENDERS_MASTERVALUE);
                if (d.typemod == 1) mono.cplex.setAnnotation(decomp, mono.cd_func[d.ncd * (t - 1) + j - 1], CPX_BENDERS_MASTERVALUE);
            }
            for (IloInt k = 1; k <= d.nf; k++){
                mono.cplex.setAnnotation(decomp, mono.f_open[d.nf * (t - 1) + k - 1], CPX_BENDERS_MASTERVALUE);
                mono.cplex.setAnnotation(decomp, mono.f_clos[d.nf * (t - 1) + k - 1], CPX_BENDERS_MASTERVALUE);
                if (d.typemod == 1 || d.typemod == 2) mono.cplex.setAnnotation(decomp, mono.f_func[d.nf * (t - 1) + k - 1], CPX_BENDERS_MASTERVALUE);
            }
            for (IloInt j = 1; j <= d.ncd; j++){
                for (IloInt k = 1; k <= d.nf; k++) mono.cplex.setAnnotation(decomp, mono.w[d.nf * d.ncd * (t - 1) + d.nf * (j - 1) + k - 1], CPX_BENDERS_MASTERVALUE);
            }
        }
    }
    else{
        //versão full
        if (typeBenders == 2) mono.cplex.setParam(IloCplex::Param::Benders::Strategy, IloCplex::BendersFull);
    }
    
    solve_model(d, mono);
    double t_final = crono.getTime();
    
    //printf (" Valor da Fo: %4f | Tempo %4f\n", mono.of, t_final);
    //print_solution_mono(d, mono);
    print_solution_mono_new(d, mono);
    print_solution_arq(d, mono, argv, typeBenders, t_final);
    
    //char nome2[250];
    char nome3[250];
    
    char stra[250] = "aux-DSA-CPLEX";
    //char strb[250] = "result-DSA-CPLEX";
    
    if (typeBenders == 1){
        strcat (stra,"-BD-Annot.txt");
        //strcat (strb,"-BD-Annot.txt");
        //strcpy(nome3, stra);
        //strcpy(nome2, strb);
    }
    else{
        if (typeBenders == 2){
            strcat (stra,"BD-Full");
            //strcat (strb,"BD-Full");
            //strcpy(nome3, stra);
            //strcpy(nome2, strb);
        }
    }
    
    if (d.typemod == 1){
        strcat (stra,"-Mod1.txt");
        //strcat (strb,"-Mod1.txt");
        strcpy(nome3, stra);
        //strcpy(nome2, strb);
    }
    
    if (d.typemod == 2){
        strcat (stra,"-Mod2.txt");
        //strcat (strb,"-Mod2.txt");
        strcpy(nome3, stra);
        //strcpy(nome2, strb);
    }
    
    if (d.typemod == 3){
        strcat (stra,"-Mod3.txt");
        //strcat (strb,"-Mod3.txt");
        strcpy(nome3, stra);
        //strcpy(nome2, strb);
    }
    
    if (d.typemod == 4){
        strcat (stra,"-Mod4.txt");
        //strcat (strb,"-Mod4.txt");
        strcpy(nome3, stra);
        //strcpy(nome2, strb);
    }
        
    /*FILE *arq;
    arq = fopen(nome2, "aw+");
    fprintf (arq,"%s \n",argv[1]);
    fprintf(arq, "\n  H |          LB        |          SUP       |              Time \n");*/
    
    int nstat = d.sup.size();
    //apenas se não abrir nó nenhum  - instâncias pequenas
    if(nstat == 0){
        //float t_final = crono.getTime();
        //fprintf (arq,"%3d | %18.4f | %18.4f | %18.4f  |   %s\n",0, mono.of, mono.of, t_final, argv[1]);
        FILE *arq3;
        arq3 = fopen(nome3, "aw+");
        fprintf (arq3,"%3d | %18.4f | %18.4f | %18.4f |   %s\n",0, mono.of, mono.of, t_final, argv[1]);
        fclose(arq3);
    }
    else{
        FILE *arq3;
        arq3 = fopen(nome3, "aw+");
        fprintf (arq3,"%3d | %18.4f | %18.4f | %18.4f |   %s\n",nstat-1, d.lb[nstat-1], d.sup[nstat-1], d.time[nstat-1], argv[1]);
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

void create_model (DAT &d, CPX_DAT &mono){
    IloEnv env = mono.env;
    mono.mod = IloModel(env); 
    mono.cplex = IloCplex(mono.mod);
    
    mono.f_open = IloNumVarArray(env, d.np * d.nf, 0.0, 1.0, ILOINT);   // openning
    mono.f_clos = IloNumVarArray(env, d.np * d.nf, 0.0, 1.0, ILOINT);   // closing
    mono.cd_open = IloNumVarArray(env, d.np * d.ncd, 0.0, 1.0, ILOINT); // openning
    mono.cd_clos = IloNumVarArray(env, d.np * d.ncd, 0.0, 1.0, ILOINT); // closing
    mono.x = IloNumVarArray(env, d.np * d.nc * d.ncd * d.nf, 0.0,+IloInfinity,ILOFLOAT);   
    mono.w = IloNumVarArray(env, d.np * d.ncd * d.nf, 0.0, 1.0, ILOINT);
    mono.constraints = IloRangeArray(env);
    mono._f_open = IloNumArray(env, d.np * d.nf);
    mono._f_clos = IloNumArray(env, d.np * d.nf);
    mono._cd_open = IloNumArray(env, d.np * d.ncd);
    mono._cd_clos = IloNumArray(env, d.np * d.ncd);

    if (d.typemod == 1){
        mono.f_func = IloNumVarArray(env, d.np * d.nf, 0.0, 1.0, ILOINT);   // functioning
        mono._f_func = IloNumArray(env, d.np * d.nf);
        mono.cd_func = IloNumVarArray(env, d.np * d.ncd, 0.0, 1.0, ILOINT); // functioning
        mono._cd_func = IloNumArray(env, d.np * d.ncd);
    }
    if (d.typemod == 2){
        mono.f_func = IloNumVarArray(env, d.np * d.nf, 0.0, 1.0, ILOINT);   // functioning
        mono._f_func = IloNumArray(env, d.np * d.nf);
        mono._w = IloNumArray(env, d.np * d.ncd * d.nf);
    }
    if (d.typemod == 3 || d.typemod == 4) mono._w = IloNumArray(env, d.np * d.ncd * d.nf);

    // ===============
    // Função objetivo do mestre
    // ===============
    if (d.typemod == 1){
        IloExpr xpfo(env);
        for (int t = 1; t <= d.np; t++){
            for (int k = 1; k <= d.nf; k++){
                xpfo += (d.cf_open[t][k] * mono.f_open[d.nf * (t - 1) + k - 1]);
                xpfo += (d.cf_clos[t][k] * mono.f_clos[d.nf * (t - 1) + k - 1]);
                xpfo += (d.cf_func[t][k] * mono.f_func[d.nf * (t - 1) + k - 1]);
            }
            for (int j = 1; j <= d.ncd; j++){
                xpfo += (d.ccd_open[t][j] * mono.cd_open[d.ncd * (t - 1) + j - 1]);
                xpfo += (d.ccd_clos[t][j] * mono.cd_clos[d.ncd * (t - 1) + j - 1]);
                xpfo += (d.ccd_func[t][j] * mono.cd_func[d.ncd * (t - 1) + j - 1]);
            }
            for(int i = 1; i <= d.nc; i++){
                for(int j = 1; j <= d.ncd; j++){
                    for(int k = 1; k <= d.nf; k++) xpfo += d.c[t][i][j][k] * mono.x[d.nc * d.nf * d.ncd * (t - 1) + d.nf * d.ncd * (i - 1) + d.nf * (j - 1) + k - 1];
                }
            }
        }
        mono.fo = IloAdd(mono.mod, IloMinimize(env, xpfo));
        xpfo.end();
    }

    if (d.typemod == 2){
        IloExpr xpfo(env);
        for (int t = 1; t <= d.np; t++){
            for (int k = 1; k <= d.nf; k++){
                xpfo += (d.cf_open[t][k] * mono.f_open[d.nf * (t - 1) + k - 1]);
                xpfo += (d.cf_clos[t][k] * mono.f_clos[d.nf * (t - 1) + k - 1]);
                xpfo += (d.cf_func[t][k] * mono.f_func[d.nf * (t - 1) + k - 1]);
            }
            for (int j = 1; j <= d.ncd; j++){
                xpfo += (d.ccd_open[t][j] * mono.cd_open[d.ncd * (t - 1) + j - 1]);
                xpfo += (d.ccd_clos[t][j] * mono.cd_clos[d.ncd * (t - 1) + j - 1]);
                for(int k = 1; k <= d.nf; k++){
                    xpfo += (d.ccd_func[t][j] * mono.w[d.nf * d.ncd * (t - 1) + d.nf * (j - 1) + k - 1]);
                }
            }
            for(int i = 1; i <= d.nc; i++){
                for(int j = 1; j <= d.ncd; j++){
                    for(int k = 1; k <= d.nf; k++) xpfo += d.c[t][i][j][k] * mono.x[d.nc * d.nf * d.ncd * (t - 1) + d.nf * d.ncd * (i - 1) + d.nf * (j - 1) + k - 1];
                }
            }
        }
        mono.fo = IloAdd(mono.mod, IloMinimize(env, xpfo));
        xpfo.end();
    }
    
    if (d.typemod == 3){
        IloExpr xpfo(env);
        for (int t = 1; t <= d.np; t++){
            for (int k = 1; k <= d.nf; k++){
                xpfo += (d.cf_open[t][k] * mono.f_open[d.nf * (t - 1) + k - 1]);
                xpfo += (d.cf_clos[t][k] * mono.f_clos[d.nf * (t - 1) + k - 1]);
                for (int r = 1; r <= t; r++) xpfo += (d.cf_func[t][k] * mono.f_open[d.nf * (r - 1) + k - 1]);
                for (int r = 2; r <= t; r++) xpfo -= (d.cf_func[t][k] * mono.f_clos[d.nf * (r - 1) + k - 1]);
            }
            for (int j = 1; j <= d.ncd; j++){
                xpfo += (d.ccd_open[t][j] * mono.cd_open[d.ncd * (t - 1) + j - 1]);
                xpfo += (d.ccd_clos[t][j] * mono.cd_clos[d.ncd * (t - 1) + j - 1]);
                for (int r = 1; r <= t; r++) xpfo += (d.ccd_func[t][j] * mono.cd_open[d.ncd * (r - 1) + j - 1]);
                for (int r = 2; r <= t; r++) xpfo -= (d.ccd_func[t][j] * mono.cd_clos[d.ncd * (r - 1) + j - 1]);
            }
            for(int i = 1; i <= d.nc; i++){
                for(int j = 1; j <= d.ncd; j++){
                    for(int k = 1; k <= d.nf; k++) xpfo += d.c[t][i][j][k] * mono.x[d.nc * d.nf * d.ncd * (t - 1) + d.nf * d.ncd * (i - 1) + d.nf * (j - 1) + k - 1];
                }
            }
        }
        mono.fo = IloAdd(mono.mod, IloMinimize(env, xpfo));
        xpfo.end();
    }
    
    if (d.typemod == 4){
        IloExpr xpfo(env);
        for (int t = 1; t <= d.np; t++){
            for (int k = 1; k <= d.nf; k++){
                xpfo += (d.cf_open[t][k] * mono.f_open[d.nf * (t - 1) + k - 1]);
                xpfo += (d.cf_clos[t][k] * mono.f_clos[d.nf * (t - 1) + k - 1]);
                for (int r = 1; r <= t; r++) xpfo += (d.cf_func[t][k] * mono.f_open[d.nf * (r - 1) + k - 1]);
                for (int r = 2; r <= t; r++) xpfo -= (d.cf_func[t][k] * mono.f_clos[d.nf * (r - 1) + k - 1]);
            }
            for (int j = 1; j <= d.ncd; j++){
                xpfo += (d.ccd_open[t][j] * mono.cd_open[d.ncd * (t - 1) + j - 1]);
                xpfo += (d.ccd_clos[t][j] * mono.cd_clos[d.ncd * (t - 1) + j - 1]);
                for(int k = 1; k <= d.nf; k++) xpfo += (d.ccd_func[t][j] * mono.w[d.nf * d.ncd * (t - 1) + d.nf * (j - 1) + k - 1]);
            }
            for(int i = 1; i <= d.nc; i++){
                for(int j = 1; j <= d.ncd; j++){
                    for(int k = 1; k <= d.nf; k++) xpfo += d.c[t][i][j][k] * mono.x[d.nc * d.nf * d.ncd * (t - 1) + d.nf * d.ncd * (i - 1) + d.nf * (j - 1) + k - 1];
                }
            }
        }
        mono.fo = IloAdd(mono.mod, IloMinimize(env, xpfo));
        xpfo.end();
    }
 
    // ===============
    // Restrições 
    // ===============
    // Alocação das fábricas
    for (int t = 1; t <= d.np; t++){
        for(int i = 1; i <= d.nc; i++){
            IloExpr r1(env);
            for (int j = 1; j <= d.ncd; j++){
                for(int k = 1; k <= d.nf; k++) r1 += mono.x[d.nc * d.nf * d.ncd * (t - 1) + d.nf * d.ncd * (i - 1) + d.nf * (j - 1) + k - 1];
            }
            mono.constraints.add(r1 == 1);
            r1.end();
        }
    }
    
    for (int t = 1; t <= d.np; t++){
        for(int i = 1; i <= d.nc; i++){
            for (int j = 1; j <= d.ncd; j++){
                for (int k = 1; k <= d.nf; k++){
                    IloExpr r2(env);
                    r2 -= mono.w[d.nf * d.ncd * (t - 1) + d.nf * (j - 1) + k - 1];
                    r2 += mono.x[d.nc * d.nf * d.ncd * (t - 1) + d.nf * d.ncd * (i - 1) + d.nf * (j - 1) + k - 1];
                    mono.constraints.add(r2 <= 0);
                    r2.end();
                }
            }
        }
    }

    if (d.typemod == 1 || d.typemod == 2){
        for (int t = 1; t <= d.np; t++){
            for(int i = 1; i <= d.nc; i++){
                for (int k = 1; k <= d.nf; k++){
                    IloExpr r3(env);
                    r3 -= mono.f_func[d.nf * (t - 1) + k - 1];
                    for(int j = 1; j <= d.ncd; j++) r3 += mono.x[d.nc * d.nf * d.ncd * (t - 1) + d.nf * d.ncd * (i - 1) + d.nf * (j - 1) + k - 1];
                    mono.constraints.add(r3 <= 0);
                    r3.end();
                }
            }
        }
        
        for (int t = 1; t <= d.np; t++){
            for(int j = 1; j <= d.ncd; j++){
                for (int k = 1; k <= d.nf; k++){
                    IloExpr r4(env);
                    r4 -= mono.f_func[d.nf * (t - 1) + k - 1];
                    r4 += mono.w[d.nf * d.ncd * (t - 1) + d.nf * (j - 1) + k - 1];
                    mono.constraints.add(r4 <= 0);
                    r4.end();            
                }
            }
        }
    }
    
    if (d.typemod == 1){
        for (int t = 1; t <= d.np; t++){
            for(int j = 1; j <= d.ncd; j++){
                IloExpr r5(env);
                r5 -= mono.cd_func[d.ncd * (t - 1) + j - 1];
                for (int k = 1; k <= d.nf; k++) r5 += mono.w[d.nf * d.ncd * (t - 1) + d.nf * (j - 1) + k - 1];
                mono.constraints.add(r5 == 0);
                r5.end();
            }
        }
    }
    
    if (d.typemod == 1 || d.typemod == 2){
        for (int t = 2; t <= d.np; t++){
            for(int k = 1; k <= d.nf; k++){
                IloExpr r6(env);
                r6 += mono.f_func[d.nf * (t - 1) + k - 1];
                r6 -= mono.f_open[d.nf * (t - 1) + k - 1];
                r6 -= mono.f_func[d.nf * (t - 2) + k - 1];
                r6 += mono.f_clos[d.nf * (t - 1) + k - 1];
                mono.constraints.add(r6 == 0);
                r6.end();
            }
        }
        
        for(int k = 1; k <= d.nf; k++){
            IloExpr r7(env);
            r7 += mono.f_func[k - 1];
            r7 -= mono.f_open[k - 1];
            mono.constraints.add(r7 == 0);
            r7.end();
        }
    }
    
    if (d.typemod == 1){
        for (int t = 2; t <= d.np; t++){
            for(int j = 1; j <= d.ncd; j++){
                IloExpr r8(env);
                r8 += mono.cd_func[d.ncd * (t - 1) + j - 1];
                r8 -= mono.cd_open[d.ncd * (t - 1) + j - 1];
                r8 -= mono.cd_func[d.ncd * (t - 2) + j - 1];
                r8 += mono.cd_clos[d.ncd * (t - 1) + j - 1];
                mono.constraints.add(r8 == 0);
                r8.end();
            }
        }

        for(int j = 1; j <= d.ncd; j++){
            IloExpr r9(env);
            r9 += mono.cd_func[j - 1];
            r9 -= mono.cd_open[j - 1];
            mono.constraints.add(r9 == 0);
            r9.end();
        }
    }

    if (d.typemod == 2 || d.typemod == 4){
        for(int j = 1; j <= d.ncd; j++){
            IloExpr r10(env);
            for (int k = 1; k <= d.nf; k++) r10 += mono.w[d.nf * (j - 1) + k - 1];
            r10 -= mono.cd_open[j - 1];
            mono.constraints.add(r10 == 0);
            r10.end();
        }

        for (int t = 2; t <= d.np; t++){
            for(int j = 1; j <= d.ncd; j++){
                IloExpr r11(env);
                r11 -= mono.cd_open[d.ncd * (t - 1) + j - 1];
                r11 += mono.cd_clos[d.ncd * (t - 1) + j - 1];
                for (int k = 1; k <= d.nf; k++){
                    r11 += mono.w[d.nf * d.ncd * (t - 1) + d.nf * (j - 1) + k - 1];
                    r11 -= mono.w[d.nf * d.ncd * (t - 2) + d.nf * (j - 1) + k - 1];
                }
                mono.constraints.add(r11 == 0);
                r11.end();
            }
        }
    }
    
    if (d.typemod == 3 || d.typemod == 4){
        for (int t = 1; t <= d.np; t++){
            for(int i = 1; i <= d.nc; i++){
                for (int k = 1; k <= d.nf; k++){
                    IloExpr r12(env);
                    for (int r = 1; r <= t; r++) r12 -= mono.f_open[d.nf * (r - 1) + k - 1];
                    for (int r = 2; r <= t; r++) r12 += mono.f_clos[d.nf * (r - 1) + k - 1];
                    for(int j = 1; j <= d.ncd; j++) r12 += mono.x[d.nc * d.nf * d.ncd * (t - 1) + d.nf * d.ncd * (i - 1) + d.nf * (j - 1) + k - 1];
                    mono.constraints.add(r12 <= 0);
                    r12.end();
                }
            }
        }
        
        for (int t = 1; t <= d.np; t++){
            for(int j = 1; j <= d.ncd; j++){
                for (int k = 1; k <= d.nf; k++){
                    IloExpr r13(env);
                    for (int r = 1; r <= t; r++) r13 -= mono.f_open[d.nf * (r - 1) + k - 1];
                    for (int r = 2; r <= t; r++) r13 += mono.f_clos[d.nf * (r - 1) + k - 1];
                    r13 += mono.w[d.nf * d.ncd * (t - 1) + d.nf * (j - 1) + k - 1];
                    mono.constraints.add(r13 <= 0);
                    r13.end();            
                }
            }
        }    
    }
    
    if (d.typemod == 3){
        for (int t = 1; t <= d.np; t++){
            for(int j = 1; j <= d.ncd; j++){
                IloExpr r14(env);
                for (int k = 1; k <= d.nf; k++) r14 += mono.w[d.nf * d.ncd * (t - 1) + d.nf * (j - 1) + k - 1];
                for (int r = 1; r <= t; r++) r14 -= mono.cd_open[d.ncd * (r - 1) + j - 1];
                for (int r = 2; r <= t; r++) r14 += mono.cd_clos[d.ncd * (r - 1) + j - 1];
                mono.constraints.add(r14 == 0);
                r14.end();
            }
        }    
    }
    
    //mono.cplex.setParam(IloCplex::Param::RandomSeed, 1);         //To define if are use the defaut seed or not
    mono.cplex.setParam(IloCplex::Threads,1);
    //mono.cplex.setWarning(env.getNullStream());
    //mono.cplex.setOut(env.getNullStream());
    mono.cplex.setParam(IloCplex::TiLim, 86400);
    mono.mod.add(mono.constraints);
}

void solve_model (DAT &d, CPX_DAT &mono){
    mono.cplex.solve();
    mono.of = (double) mono.cplex.getObjValue();
    mono.cplex.getValues(mono._f_open, mono.f_open);
    mono.cplex.getValues(mono._f_clos, mono.f_clos);
    mono.cplex.getValues(mono._cd_open, mono.cd_open);    
    mono.cplex.getValues(mono._cd_clos, mono.cd_clos);    
    if (d.typemod == 1){
        mono.cplex.getValues(mono._f_func, mono.f_func);
        mono.cplex.getValues(mono._cd_func, mono.cd_func);
    }
    if (d.typemod == 2){
        mono.cplex.getValues(mono._f_func, mono.f_func);
        mono.cplex.getValues(mono._w, mono.w);
    }
    if (d.typemod == 3 || d.typemod == 4) mono.cplex.getValues(mono._w, mono.w);
}

// ========================================================
// Imprime a Solução final
// ========================================================
void print_solution_mono(DAT &d, CPX_DAT &mono){
    for (int t = 1; t <= d.np; t++){
        printf(" Openned Facility First Level:\n");
        for (int k = 1; k <= d.nf; k++){
            if (mono._f_open[d.nf * (t - 1) + k - 1] > -0.0) printf("  %4d ", k);
        }
        printf("\n");
        
        printf(" Openned Facility Second Level :\n");
        for (int j = 1; j <= d.ncd; j++){
            if (mono._cd_open[d.ncd * (t - 1) + j - 1] > -0.0) printf("  %4d ", j);
        }
        printf("\n");
       
        printf(" Closed Facility First Level:\n");
        for (int k = 1; k <= d.nf; k++){
            if (mono._f_clos[d.nf * (t - 1) + k - 1] > -0.0) printf("  %4d ", k);
        }
        printf("\n");
        
        printf(" Closed Facility Second Level :\n");
        for (int j = 1; j <= d.ncd; j++){
            if (mono._cd_clos[d.ncd * (t - 1) + j - 1] > -0.0) printf("  %4d ", j);
        }
        printf("\n");
    
        if (d.typemod == 1){
            printf(" Functioning Facility First Level:\n");
            for (int k = 1; k <= d.nf; k++){
                if (mono._f_func[d.nf * (t - 1) + k - 1] > -0.0) printf("  %4d ", k);
            }
            printf("\n");
            
            printf(" Functioning Facility Second Level :\n");
            for (int j = 1; j <= d.ncd; j++){
                if (mono._cd_func[d.ncd * (t - 1) + j - 1] > -0.0) printf("  %4d ", j);
            }
            printf("\n");
        }
        
        if (d.typemod == 2){
            printf(" Functioning Facility First Level:\n");
            for (int k = 1; k <= d.nf; k++){
                if (mono._f_func[d.nf * (t - 1) + k - 1] > -0.0) printf("  %4d ", k);
            }
            printf("\n");
            
            printf(" Functioning Facility Second Level :\n");
            for (int j = 1; j <= d.ncd; j++){
                double conta = 0;
                for (int k = 1; k <= d.nf; k++){
                    if (mono._w[d.nf * d.ncd * (t - 1) + d.nf * (j - 1) + k - 1] > -0.0 && conta == 0){
                        printf("  %4d ", j);
                        conta = 1;
                    }
                }
            }
            printf("\n");
        }
        
        if (d.typemod == 3 || d.typemod == 4){
            printf(" Functioning Facility First Level:\n");
            for (int k = 1; k <= d.nf; k++){
                double conta = 0;
                for (int j = 1; j <= d.ncd; j++){
                    if (mono._w[d.nf * d.ncd * (t - 1) + d.nf * (j - 1) + k - 1] > -0.0 && conta == 0){
                        printf("  %4d ", k);
                        conta = 1;
                    }
                }
            }
            printf("\n");
            
            printf(" Functioning Facility Second Level :\n");
            for (int j = 1; j <= d.ncd; j++){
                double conta = 0;
                for (int k = 1; k <= d.nf; k++){
                    if (mono._w[d.nf * d.ncd * (t - 1) + d.nf * (j - 1) + k - 1] > -0.0 && conta == 0){
                        printf("  %4d ", j);
                        conta = 1;
                    }
                }
            }
            printf("\n");
        }
        printf("\n ===================================================== \n");
    }
}

// ========================================================
// Print Final Solution
// ========================================================
void print_solution_mono_new(DAT &d, CPX_DAT &mono){
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
            if (mono._f_open[d.nf * (t - 1) + k - 1] > 0.9){
                verifica = 1;
                t = d.np + 1;
            }
        }
        if (verifica == 1){
            if(k>99) printf(" %d  ", k); //100 a 999 
            else{
                if(k<10) printf("   %d  ", k); //1 a 9
                else printf("  %d  ", k); //10 a 99
            }
            for (int t = 1; t <= d.np; t++){
                //open
                if (mono._f_open[d.nf * (t - 1) + k - 1] > 0.9) printf("|  %d", 1);
                else printf("|   ");
                //functioning
                if (d.typemod == 1 || d.typemod == 2){
                    if (mono._f_func[d.nf * (t - 1) + k - 1] > 0.9) printf("  %d", 1);
                    else printf("   ");
                }
                else{
                    double val = 0;
                    for (int r = 1; r <= t; r++) val += mono._f_open[d.nf * (r - 1) + k - 1];
                    for (int r = 2; r <= t; r++) val -= mono._f_clos[d.nf * (r - 1) + k - 1];
                    if (val > 0.9) printf("  %d", 1);
                    else printf("   ");
                }
                //closing
                if (mono._f_clos[d.nf * (t - 1) + k - 1] > 0.9) printf("  %d  ", 1);
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
            if (mono._cd_open[d.ncd * (t - 1) + j - 1] > 0.9){
                verifica = 1;
                t = d.np + 1;
            }
        }
        if (verifica == 1){
            if(j>99) printf(" %d  ", j); //100 a 999 
            else{
                if(j<10) printf("   %d  ", j); //1 a 9
                else printf("  %d  ", j); //10 a 99
            }
            for (int t = 1; t <= d.np; t++){
                //open
                if (mono._cd_open[d.ncd * (t - 1) + j - 1] > 0.9) printf("|  %d", 1);
                else printf("|   ");
                //functioning
                if (d.typemod == 1){
                    if (mono._cd_func[d.ncd * (t - 1) + j - 1] > 0.9) printf("  %d", 1);
                    else printf("   ");
                }
                else{
                    double val = 0;
                    for (int r = 1; r <= t; r++) val += mono._cd_open[d.ncd * (r - 1) + j - 1];
                    for (int r = 2; r <= t; r++) val -= mono._cd_clos[d.ncd * (r - 1) + j - 1];
                    if (val > 0.9) printf("  %d", 1);
                    else printf("   ");
                }
                //closing
                if (mono._cd_clos[d.ncd * (t - 1) + j - 1] > 0.9) printf("  %d  ", 1);
                else printf("     ");
            }
            printf("\n");
        }
    }
    printf("================================================================= \n");
    printf("Valor da FO: %f\n", mono.of);
    printf("================================================================= \n");
}

// ========================================================
// Salva a solução final em um arquivo
// ========================================================
void print_solution_arq(DAT &d, CPX_DAT &mono, char *argv[], int typeBenders, double t_final){
    char nome[250];
    char stra[250] = "sol-DSA-CPLEX";
    if (typeBenders == 1) strcat (stra,"-BD-Annot.txt");
    else{
        if (typeBenders == 2) strcat (stra,"BD-Full");
    }
    if (d.typemod == 1){
        strcat (stra,"-Mod1.txt");
        strcpy(nome, stra);
    }
    if (d.typemod == 2){
        strcat (stra,"-Mod2.txt");
        strcpy(nome, stra);
    }
    if (d.typemod == 3){
        strcat (stra,"-Mod3.txt");
        strcpy(nome, stra);
    }
    if (d.typemod == 4){
        strcat (stra,"-Mod4.txt");
        strcpy(nome, stra);
    }
            
    FILE *arq;
    arq = fopen(nome, "aw+");
    fprintf (arq,"================================================================= \n");
    fprintf (arq,"Instance: %s\n", argv[1]);
    fprintf (arq,"Objective Function: %f\n", mono.of);
    fprintf (arq,"Time: %f\n", t_final);
    
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
            if (mono._f_open[d.nf * (t - 1) + k - 1] > 0.9){
                verifica = 1;
                t = d.np + 1;
            }
        }
        if (verifica == 1){
            if(k>99) fprintf (arq," %d  ", k); //100 a 999 
            else{
                if(k<10) fprintf (arq,"   %d  ", k); //1 a 9
                else fprintf (arq,"  %d  ", k); //10 a 99
            }
            for (int t = 1; t <= d.np; t++){
                //open
                if (mono._f_open[d.nf * (t - 1) + k - 1] > 0.9) fprintf (arq,"|  %d", 1);
                else fprintf (arq,"|   ");
                //functioning
                if (d.typemod == 1 || d.typemod == 2){
                    if (mono._f_func[d.nf * (t - 1) + k - 1] > 0.9) fprintf (arq,"  %d", 1);
                    else fprintf (arq,"   ");
                }
                else{
                    double val = 0;
                    for (int r = 1; r <= t; r++) val += mono._f_open[d.nf * (r - 1) + k - 1];
                    for (int r = 2; r <= t; r++) val -= mono._f_clos[d.nf * (r - 1) + k - 1];
                    if (val > 0.9) fprintf (arq,"  %d", 1);
                    else fprintf (arq,"   ");
                }
                //closing
                if (mono._f_clos[d.nf * (t - 1) + k - 1] > 0.9) fprintf (arq,"  %d  ", 1);
                else fprintf (arq,"     ");
            }
            fprintf (arq,"\n");
        }
    }
    
    fprintf (arq,"\n\n");
    fprintf (arq,"Second Level:\n");
    fprintf (arq,"      ");
    for (int t = 1; t <= d.np; t++) fprintf (arq,"|    t=%d    ", t);
    fprintf (arq,"\n Fac  ");
    for (int t = 1; t <= d.np; t++) fprintf (arq,"|  O  F  C  ");
    fprintf (arq,"\n");
    
    for (int j = 1; j <= d.ncd; j++){
        int verifica = 0;
        for (int t = 1; t <= d.np; t++){
            if (mono._cd_open[d.ncd * (t - 1) + j - 1] > 0.9){
                verifica = 1;
                t = d.np + 1;
            }
        }
        if (verifica == 1){
            if(j>99) fprintf (arq," %d  ", j); //100 a 999 
            else{
                if(j<10) fprintf (arq,"   %d  ", j); //1 a 9
                else fprintf (arq,"  %d  ", j); //10 a 99
            }
            for (int t = 1; t <= d.np; t++){
                //open
                if (mono._cd_open[d.ncd * (t - 1) + j - 1] > 0.9) fprintf (arq,"|  %d", 1);
                else fprintf (arq,"|   ");
                //functioning
                if (d.typemod == 1){
                    if (mono._cd_func[d.ncd * (t - 1) + j - 1] > 0.9) fprintf (arq,"  %d", 1);
                    else fprintf (arq,"   ");
                }
                else{
                    double val = 0;
                    for (int r = 1; r <= t; r++) val += mono._cd_open[d.ncd * (r - 1) + j - 1];
                    for (int r = 2; r <= t; r++) val -= mono._cd_clos[d.ncd * (r - 1) + j - 1];
                    if (val > 0.9) fprintf (arq,"  %d", 1);
                    else fprintf (arq,"   ");
                }
                //closing
                if (mono._cd_clos[d.ncd * (t - 1) + j - 1] > 0.9) fprintf (arq,"  %d  ", 1);
                else fprintf (arq,"     ");
            }
            fprintf (arq,"\n");
        }
    }
    fprintf (arq,"================================================================= \n");
    fprintf (arq,"\n\n\n");
    fclose(arq);
}
