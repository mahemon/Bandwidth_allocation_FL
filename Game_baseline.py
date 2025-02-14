import math
import copy
import random
import numpy as np

class Game_base:
    def __init__(self, n_fs, n_es):
        self.inf = 999
        self.n_es = n_es
        self.n_fs = n_fs
        self.p = 1  # unit price of bandwidth at each edge server
        self.x_i_j = [[0 for i in range(self.n_es)] for j in range(self.n_fs)]  # 2D list col ES and row FS # units of bandwidth allocated to S_i by E_j
        self.rk_i = [0 for i in range(0, n_fs)]  # rank of s_i
        self.rr_i_j = []
        self.rb_j = []
        self.c_i_j = []
        self.u_i = []
        self.frac = []
        self.num_user_in_fs = 0
        self.init_rb_j = []
        self.user_ES_j = []
        self.total_user_each_es = 0

    def random_sum_of_n(self, summation, n):
        rangee = math.floor(summation / n) + n
        while True:
            pick = np.random.choice(range(1, rangee), n)  # n -> n_es with repeat
            if sum(pick) <= summation and sum(pick) >= (summation - n):
                break
        return pick

    def cal_rr_i_j(self):
        for i in range(0, self.n_fs):
            temp = []
            for j in range(0, self.n_es):
                temp.append(min(math.floor(self.rb_j[j] / self.u_i[i]), self.c_i_j[i][j]) * self.u_i[i])
            self.rr_i_j.append(temp)

    def profit(self, u_i, frac, num_user_in_fs, alpha, beta, total_user_each_es):
        self.frac = frac
        self.num_user_in_fs = num_user_in_fs
        self.u_i = [u_i for i in range(0, self.n_fs)]
        self.total_user_each_es = total_user_each_es
        fs_alpha_list, es_beta_list = [], []
        es_beta_list = {}
        beta_es = 0
        if alpha > 0 and alpha <= 1:
            alph_fs = math.floor(alpha * self.n_fs)
            beta_es = math.floor(beta * self.n_es)
            fs_alpha_list = [i for i in range(0, alph_fs)]

        for i in range(0, self.n_fs):
            if i in fs_alpha_list:
                temp = list(self.random_sum_of_n(self.num_user_in_fs, beta_es))
                for i in range(beta_es, self.n_es):
                    temp.append(0)
                self.c_i_j.append(temp) # beta -> no of es server
            else:
                self.c_i_j.append(list(self.random_sum_of_n(self.num_user_in_fs, self.n_es)))
        for j in range(0, self.n_es):
            self.user_ES_j.append(self.total_user_each_es)
        # Bandwidth
        for j in range(0, self.n_es):
            self.rb_j.append(self.user_ES_j[j])

        self.init_rb_j = copy.deepcopy(self.rb_j)
        self.rr_i_j = self.c_i_j
        req_es = [ sum(x) for x in zip(*self.rr_i_j) ]
        request_to_each_es = np.sum(self.c_i_j, axis=0)
        for es in range(self.n_es):
            if request_to_each_es[es] > self.rb_j[es]:
                t_reduce = 0
                for fs in range(self.n_fs):
                    self.x_i_j[fs][es] = round((self.c_i_j[fs][es] * self.rb_j[es]) / request_to_each_es[es])
                    t_reduce = t_reduce + self.x_i_j[fs][es]
                self.rb_j[es] = self.rb_j[es] - t_reduce

                if self.rb_j[es] > 0:
                    for fs in range(self.n_fs):
                        if self.c_i_j[fs][es] > 0 and self.rb_j[es] > 0:
                            self.x_i_j[fs][es] = self.x_i_j[fs][es] + 1
                            self.rb_j[es] = self.rb_j[es] - 1
                            if self.rb_j[es] == 0:
                                break;

                if self.rb_j[es] < 0:
                    for fs in range(self.n_fs):
                        if self.rb_j[es] < 0:
                            if self.x_i_j[fs][es] > 0:
                                self.x_i_j[fs][es] = self.x_i_j[fs][es] - 1
                                self.rb_j[es] = self.rb_j[es] + 1
                            if self.rb_j[es] == 0:
                                break;

            else:
                for fs in range(self.n_fs):
                    self.x_i_j[fs][es] = self.rr_i_j[fs][es]
                    self.rb_j[es] = self.rb_j[es] - self.x_i_j[fs][es]
        return self.x_i_j, self.rb_j, self.rr_i_j

if __name__ == '__main__':

    u_i = 1
    frac = 1
    num_user_in_fs = 100
    total_user_each_es = 10
    num_servers = 5
    num_edge_server = 5
    alpha_list = [0.2, 0.4, 0.6, 0.8, 0.0]
    beta_list = [0.2, 0.4, 0.6, 0.8, 1.0]

    for alpha in alpha_list:
        for beta in beta_list:
            obj2 = Game_base(num_servers, num_edge_server)
            print("~" * 100)
            print("alpha: ", alpha, ", beta: ", beta)
            xij, remain_rb_j, assinged_rr_i_j = obj2.profit(u_i, frac, num_user_in_fs, alpha, beta, total_user_each_es)  # rb_j, c_i_j, u_i
            print(f'x_i_j: {xij}')
            fs_ed = []
            for i in range(len(xij)):
                fs_ed.append(sum(xij[i]))
            print(f'ED in FS: {fs_ed}\n')
            print(f'remain_rb_j: {remain_rb_j}\n')
