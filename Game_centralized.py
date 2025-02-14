import math
import copy
import random
import numpy as np

class Game_centralized:
    def __init__(self, n_fs, n_es):
        self.inf = 999
        self.n_es = n_es
        self.n_fs = n_fs
        self.p = 1  # unit price of bandwidth at each edge server
        self.x_i_j = [[0 for i in range(self.n_es)] for j in
                      range(self.n_fs)]  # 2D list col ES and row FS # units of bandwidth allocated to S_i by E_j
        self.rk_i = [0 for i in range(0, n_fs)]  # rank of s_i

        self.rr_i_j = []
        self.rb_j = []
        self.c_i_j = []
        self.f_i = []
        self.u_i = []
        self.frac = []
        self.num_user_in_fs = 0
        self.init_rb_j = []
        self.total_user_each_es = 0

    def get_minIndex(self, inputlist):
        # get the minimum value in the list
        min_value = min(inputlist)

        # return the index of minimum value
        min_index = inputlist.index(min_value)

        return min_index

    def random_sum_of_n(self, summation, n):
        rangee = math.floor(summation / n) + n
        while True:
            pick = np.random.choice(range(1, rangee), n)  # n -> n_es with repeat
            if sum(pick) <= summation and sum(pick) >= (summation - n):
                break
        return pick

    def get_maxIndex(self, inputlist):
        # get the minimum value in the list
        max_value = max(inputlist)
        if max_value == -999:
            return -999
        # return the index of minimum value
        max_index = inputlist.index(max_value)
        return max_index

    def cal_rr_i_j(self):
        for i in range(0, self.n_fs):
            temp = []
            for j in range(0, self.n_es):
                temp.append(min(math.floor(self.rb_j[j] / self.u_i[i]), self.c_i_j[i][j]) * self.u_i[i])
            self.rr_i_j.append(temp)
        print("start rr_i_j:", self.rr_i_j)

    def bandwidth_Heterogeneity(self, u_i, delta):
        min_band = u_i
        max_band = self.n_fs
        n_server = self.n_fs
        delta_fs_list, u_i_list = [], []
        k = 0
        fs_list = [i for i in range(0, n_server)]
        u_i_list = [0 for i in range(0, n_server)]
        if delta >= 0 and delta <= 1:
            k = math.floor(delta * n_server)
            if delta == 1:
                k = k - 1
            for i in range(k):
                delta_fs_list.append(fs_list[n_server - (i + 1)])
        cnt = 0
        for i in fs_list:
            if i in delta_fs_list:
                cnt = cnt + 1
                u_i_list[i] = math.floor(min_band + ((cnt * (max_band - min_band)) / k))
            else:
                u_i_list[i] = min_band
        a = copy.deepcopy(u_i_list)
        a.reverse()
        return a

    def argmax_j(self, i_start):
        j_val = []
        j_prime = []
        for j in range(0, self.n_es):
            if self.x_i_j[i_start][j] < self.c_i_j[i_start][j]: # and self.rb_j[j] > 0
                j_prime.append(j)

        not_inf_fs_rank = [i for i, x in enumerate(self.rk_i) if
                           x < self.inf]  # all occurrences of an element in a list
        sum_of_not_inf_fs_and_j_prime = []
        sum_of_fund_not_inf_fs = 0

        for i in not_inf_fs_rank:
            # Σ_rk_i < inf and then Σ_j': rr_i_j'
            tmp = 0
            for j in j_prime:
                tmp = tmp + self.rr_i_j[i][j]
            sum_of_not_inf_fs_and_j_prime.append(tmp)

            # Σ_rk_i < inf : f_i
            sum_of_fund_not_inf_fs = sum_of_fund_not_inf_fs + self.f_i[i]

        for j in range(0, self.n_es):
            if j in j_prime:
                remaining_rb_j = self.rb_j[j] - self.u_i[i_start]
                a_i, b_i, summation = 0, 0, 0
                index = 0
                for i in not_inf_fs_rank:
                    if sum_of_not_inf_fs_and_j_prime[index] > 0 and sum_of_fund_not_inf_fs > 0:
                        a_i = (self.rr_i_j[i][j]) / sum_of_not_inf_fs_and_j_prime[index]
                        b_i = self.f_i[i] / sum_of_fund_not_inf_fs
                        summation = summation + (a_i * b_i)
                    index = index + 1
                if summation > 0:
                    j_val.append(remaining_rb_j / summation)
                else:
                    j_val.append(-999)
            else:
                j_val.append(-999)
        return self.get_maxIndex(j_val)

    def profit(self, u_i, frac, num_user_in_fs, f_i_list, alpha, beta, delta, total_user_each_es):
        self.frac = frac
        self.num_user_in_fs = num_user_in_fs
        self.total_user_each_es = total_user_each_es
        self.u_i = self.bandwidth_Heterogeneity(u_i, delta)
        self.f_i = list(f_i_list)
        fs_alpha_list, es_beta_list = [], []
        beta_es = 0
        if 0 < alpha <= 1:
            alph_fs = math.floor(alpha * self.n_fs)
            beta_es = math.floor(beta * self.n_es)
            fs_alpha_list = [i for i in range(0, alph_fs)]

        for i in range(0, self.n_fs):
            if i in fs_alpha_list:
                temp = list(self.random_sum_of_n(self.num_user_in_fs, beta_es))
                for i in range(beta_es, self.n_es):
                    temp.append(0)
                self.c_i_j.append(temp)  # beta -> no of es server
            else:
                self.c_i_j.append(list(self.random_sum_of_n(self.num_user_in_fs, self.n_es)))
        
        for j in range(0, self.n_es):
            self.rb_j.append(self.total_user_each_es)
        self.init_rb_j = copy.deepcopy(self.rb_j)
        self.cal_rr_i_j()
        while min(self.rk_i) < self.inf:
            i_start = self.get_minIndex(self.rk_i)
            if max(self.rb_j) < self.u_i[i_start]:
                self.rk_i[i_start] = self.inf
            else:
                j_start = self.argmax_j(i_start)
                if j_start == -999:
                    break

                if self.rb_j[j_start] == 0:
                    self.rk_i[i_start] = self.inf
                    continue

                self.rr_i_j[i_start][j_start] = self.rr_i_j[i_start][j_start] - self.u_i[i_start]

                self.rb_j[j_start] = self.rb_j[j_start] - self.u_i[i_start]

                self.x_i_j[i_start][j_start] = self.x_i_j[i_start][j_start] + self.u_i[i_start]

                if sum(self.x_i_j[i_start]) > ((sum(self.c_i_j[i_start]) - 1) * self.u_i[i_start]):
                    self.rk_i[i_start] = self.inf
                else:
                    self.rk_i[i_start] = (sum(self.x_i_j[i_start]) / self.f_i[i_start])
        unit_price = []
        for i in range(0, self.n_fs):
            unit_price.append(self.f_i[i] / sum(self.x_i_j[i]))

        print(f'End rr_i_j:   {self.rr_i_j}')
        print(f'unit price: {unit_price}')
        print(f'ending --- rk: {self.rk_i}')
        for ii in range(len(self.rk_i)):
            print(self.rk_i[ii])
        print(f'c_i_j: {self.c_i_j}')
        return min(unit_price), self.x_i_j, self.rb_j, self.rr_i_j

if __name__ == '__main__':
    num_user_in_fs = 200
    total_user_each_es = 10
    num_servers = 5
    num_edge_server = 5
    delta_list = [0.0]
    u_i = 1
    frac = 1

    alpha_list = [0.2, 0.4, 0.6, 0.8, 0.0]
    beta_list = [0.2, 0.4, 0.6, 0.8, 1.0]
    fs_list = [0.50 for i in range(0, num_servers)]

    for a in alpha_list:
        for b in beta_list:
            for d in delta_list:
                obj = Game_centralized(num_servers, num_edge_server)
                print("~" * 100)
                print("delta:", d)
                u_price, xij, remain_rb_j, assigned_rr_i_j = obj.profit(u_i, frac, num_user_in_fs, fs_list, a, b, d,
                                                                        total_user_each_es)  # rb_j, c_i_j, f_i, u_i

                print(f'Acquired clients by FS from each ES: {xij}')
                fs_ed = []
                rem_fund = []
                used_fund = []
                for i in range(len(xij)):
                    fs_ed.append(sum(xij[i]))
                    rem_fund.append(fs_list[i] - (sum(xij[i]) * u_price))
                    used_fund.append((sum(xij[i]) * u_price))
                print(f'Acquired total clients by FS:        {fs_ed}')
                print(f'remain_rb_j in each ES:              {remain_rb_j}\n')
                print("Unit_price:  ", u_price)
                print(f'fund in fs:   {fs_list}')
                print("used_fund:    ", used_fund)
                print("remain_fund:  ", rem_fund, "\n")
                print("c_i_j: ", obj.c_i_j)
