import subprocess as sb
#function to check all the working module is available or not
def need_packages(rq_pckg:list[str,]):
    inst_pckgs = [pckg.strip()
                 for pckg in sb.check_output(['pip', 'freeze'],
                                              universal_newlines=True)]
    miss_pckgs = [pckg for pckg in rq_pckg 
                  if pckg not in inst_pckgs]
    if miss_pckgs :
        print(f"Missing packages: {miss_pckgs}")
        sb.check_call(['pip', 'install'] + miss_pckgs)
    else:
        print(f'All required packages are installed: {inst_pckgs}')

# run the next line when running the scrpt for first time 
# need_packages(['sympy', 'numpy', 'matplotlib'])

import sympy as SY 
import numpy as np 
# import matplotlib.pyplot as plt
SY.init_printing()
D = SY.diff

class GTR:
    def __init__(self, metric:str) -> None:
        #initiate the metric
        self.metric = SY.sympify(metric)
        self.co_ords = self.sym_qsort()
        self.dimension = len(self.co_ords)
        self.g_cov = SY.Matrix(self.cov_mtr_tns(self.metric,
                                                self.co_ords))
        self.g = SY.det(self.g_cov)
        self.g_con = SY.Matrix(self.con_mtr_tns(self.g_cov))
        self.christ = self.Christoffel(self.g_cov, self.g_con,
                                       self.co_ords)
        self.rieman = self.Riemann_tensor(self.christ, self.co_ords)
        self.parallel = self.par_trans(self.christ, self.co_ords)
        self.geo_eq = self.geodesics(self.christ, self.co_ords)
        self.soln = self.solve_geodesic(self.geo_eq, self.co_ords)
        self.cov_rieman = self.R_ijkl(self.rieman, 
                                      self.g_cov, self.co_ords)

    # determining the co-ordinates
    def get_co_ords(self, metric:SY.core.add.Add):
        d = SY.Function('d')
        ind = {list(i.atoms())[0]:i 
            for i in list(metric.atoms(d))}
        return list(ind.keys())

    # sorting the symbols in order
    def sym_qsort(self, arr:list["symbols"] = None) -> list["symbols"]:
        if arr is None:
            arr = self.get_co_ords(self.metric)
        if len(arr) > 1:
            left , right = [],[]
            for i in arr :
                if i.compare(arr[0]) == 1:
                    right.append(i)
                elif i.compare(arr[0]) == -1:
                    left.append(i)
            return self.sym_qsort(left)+[arr[0]]+self.sym_qsort(right)
        else :
            return arr

    # creating the covariant metric tensor
    def cov_mtr_tns(self, mtr:SY.core.add.Add, 
                    dims:list["symbols"]):
        d = SY.Function('d')
        m = [[D(mtr, d(i),2)/2 if i==j 
            else D(mtr, d(i),d(j)) 
            for i in dims] for j in dims]
        return m

    # creating the contravariant metric tensor
    def con_mtr_tns(self, cov_g : list[list['symbols']]):
        return SY.Inverse(cov_g).doit()
    
    # calculating the Christoffel symbol components
    def christ_i_jk(self, cov:"SY.Matrix",con :"SY.Matrix",
                    X :list['symbols'],
                i : int, j:int, k:int):
        cov = SY.matrix2numpy(cov)
        con = SY.matrix2numpy(con)
        c = [con[i,m]*(D(cov[m,j],X[k]) + D(cov[m,k],X[j])
                        - D(cov[k,j],X[m]))/2 
                        for m in range(len(X))]
        return sum(c)

    # calculating  the full Christoffel symbol matrix
    def Christoffel(self, cov: 'SY.Matrix', con:'SY.Matrix',
                    co_ord:list['symbols']):
        c = [[[self.christ_i_jk(cov,con,co_ord,i,j,k)
                for k in range(len(co_ord))]
                for j in range(len(co_ord))]
                for i in range(len(co_ord))]
        return c

    # calculating the Riemann curvature tensor components
    def R_i_jkl(self, i:int, j:int, k:int, l:int,
                crist:list['symbols'], X:list['symbols']):
        r = sum([crist[i][k][m]*crist[m][j][l] - 
                crist[i][l][m]*crist[m][k][j] 
                for m in range(len(X))])
        return D(crist[i][j][l], X[k]) - D(crist[i][j][k], X[l]) + r

    #  calculating the Riemann curvature tensor
    def Riemann_tensor(self,crist:list[list[list['symbols']]],
                    co_ord:list['symbols']):
        r = [[[[self.R_i_jkl(i,j,k,l,crist,co_ord)
                for l in range(len(co_ord))]
                for k in range(len(co_ord))]
                for j in range(len(co_ord))]
                for i in range(len(co_ord))]
        return r

    # calculating parallel transport of a vector
    def par_trans(self, crist: list[list[list['symbols']]],
                co_ord:list['symbols'],
                A:list['symbols']=None):
        d = SY.Function('d')
        if A is None:
            A = [SY.Symbol(f'A^{i}')
                for i in range(len(co_ord))]
        pts = [sum([crist[i][j][k]*A[j]*d(co_ord[k]) 
            for k in range(len(co_ord))
            for j in range(len(co_ord))])
            for  i in range(len(crist))]
        return pts

    # calculating geodesic equations
    def geodesics(self, crist:list[list[list['symbols']]],
                co_ord:list['symbols'],
                s:'SY.Symbol'= SY.Symbol('s')):
        Der = SY.Derivative
        eqs = [SY.Eq( Der(co_ord[i],s,2) , -
                    sum([crist[i][j][k]*Der(co_ord[j],s)*Der(co_ord[k],s)
                        for k in range(len(co_ord))
                        for j in range(len(co_ord))]))
                        for i in range(len(co_ord))]
        return eqs
    
    #trying to solve the geodesic equations
    def solve_geodesic(self, 
                       eqn : list[SY.core.relational.Equality],
                       co_ord: list['SY.symbols'],
                       s : 'SY.symbols' = SY.S('s')):
        fn_list = [SY.Function(f'{v}') for v in co_ord]
        sub_list = [(co_ord[i], fn_list[i](s)) 
                    for i in range(len(co_ord))]
        eqn = [e.subs(sub_list) for e in eqn]
        co_ord = [v.subs(sub_list) for v in co_ord]
        try:
            soln = SY.dsolve(eqn, co_ord)
        except NotImplementedError:
            return ("Could not find solution using sympy")
        else:
            return soln
    
    # fully covariant Riemannian tensor
    def R_ijkl(self, R_i_jkl:list[list[list[list['symbols']]]],
               gcov :'SY.Matrix',
               co_ord:list['symbols']):
        gcov = SY.matrix2numpy(gcov)
        r = [[[[
            sum(
                [gcov[i,m]*R_i_jkl[m][j][k][l] 
                for m in range(len(co_ord))] )
             for l in range(len(co_ord)) ]
             for k in range(len(co_ord)) ]
             for j in range(len(co_ord)) ]
             for i in range(len(co_ord)) ]
        return r

# testing for a metric
# metric should be given as a string and
# for exact differentials(i.e, dx, dxdy, dr,... etc) use d as a function
# like dx -> d(x), dy^2 ->  d(y)**2, ...etc
if  __name__ == "__main__":
    gtr = GTR('d(x)**2 + (sin(x)*d(t))**2')
    print(gtr.cov_rieman)
