import math
import copy
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import time
import matplotlib.pyplot as plt
from scipy.special import logsumexp

class Game:
    def __init__(self, n_fs, n_es):
        self.inf = 999
        self.n_es = n_es
        self.n_fs = n_fs
        self.p = 1  # unit price of bandwidth at each edge server
        self.x_i_j = [[0 for i in range(self.n_es)] for j in
                      range(self.n_fs)]  # 2D list col ES and row FS # units of bandwidth allocated to S_i by E_j
        self.rk_i = [0 for i in range(0, n_fs)]  # rank of s_i
        self.rb_j = []
        self.c_i_j = []
        self.cr_i_j = []
        self.cr_i_j_prev = []
        self.f_i = []
        self.u_i = []
        self.threadhold = 0
        self.frac = []
        self.num_user_in_fs = 0
        self.init_rb_j = []
        self.total_user_each_es = 0
        # Create empty DataFrame
        self.df_demand = pd.DataFrame(columns=['sequence_id', 'server_id', 'edge_id', 'demand','IsPred'])
        self.df_price = pd.DataFrame(columns=['sequence_id', 'server_id', 'edge_id', 'demand', 'price', 'IncTimes'])
        self.unit_price = 0.0
        self.n_round = 0

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
                # print(f'n: {n}, sum(pick): {sum(pick)}')
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

    def convert_to_integer(self, number):
        integer_part = int(number)  # Extract the integer part of the number
        decimal_part = number - integer_part  # Extract the decimal part of the number

        if decimal_part >= 0.50:
            return integer_part + 1
        else:
            return integer_part

    # def price_dataset
    def prep_es_price_dataset(self, fs_i, es_j, demand, price =-1, IncTimes =0.0):
        if price == -1:  # prep the dataset first time
            subset = self.df_price[(self.df_price['server_id'] == fs_i) & (self.df_price['edge_id'] == es_j)]
            n_of_previous_req = len(subset)
            # Add a new row
            new_row = {'sequence_id': n_of_previous_req + 1, 'server_id': fs_i, 'edge_id': es_j, 'demand': demand, 'price': price, 'IncTimes': IncTimes}
            self.df_price.loc[len(self.df_price)] = new_row
        # update price
        else:
            subset = self.df_price[(self.df_price['server_id'] == fs_i) & (self.df_price['edge_id'] == es_j)]
            last_sequence_id = max(subset['sequence_id'])

            if last_sequence_id == 1:
                 self.df_price.loc[(self.df_price['server_id'] == fs_i) & (self.df_price['edge_id'] == es_j) & (self.df_price['sequence_id'] == last_sequence_id), ['price', 'IncTimes']] = [price, 0]
            else:
                subset_last_seq_row = subset[(subset['sequence_id'] == last_sequence_id)]
                count_IncTimes = subset_last_seq_row['IncTimes'].iloc[0]
                prev_price = subset_last_seq_row['price'].iloc[0]

                if prev_price < price:
                    count_IncTimes = count_IncTimes + 1
                else:
                    count_IncTimes = 0

                self.df_price.loc[(self.df_price['server_id'] == fs_i) & (self.df_price['edge_id'] == es_j) & (
                            self.df_price['sequence_id'] == last_sequence_id), ['price', 'IncTimes']] = [price, count_IncTimes]

                if count_IncTimes > 2:
                    return True  # remove this E_j from current S_i; third time in a row price incr
        return False

    def prep_fs_demand_dataset(self, fs_i, es_j, demand, IsPred):
        if IsPred == 2:  # update the pred value with actual deman value
            subset = self.df_demand[(self.df_demand['server_id'] == fs_i) & (self.df_demand['edge_id'] == es_j) & (self.df_demand['IsPred'] == 1)]
            last_sequence_id = max(subset['sequence_id'])
            self.df_demand.loc[(self.df_demand['fs_i'] == fs_i) & (self.df_demand['es_j'] == es_j) & (self.df_demand['sequence_id'] == last_sequence_id), ['demand']] = [demand]

        else:  # prep the dataset
            subset = self.df_demand[(self.df_demand['server_id'] == fs_i) & (self.df_demand['edge_id'] == es_j)]
            n_of_previous_req = len(subset)
            # Add a new row
            new_row = {'sequence_id': n_of_previous_req + 1, 'server_id': fs_i, 'edge_id': es_j, 'demand': demand, 'IsPred': IsPred}
            self.df_demand.loc[len(self.df_demand)] = new_row

    def pred_fs_demand(self, es_j, fs_i):
        # Fit a linear regression model for each customer-seller pair

        # Filter data for this customer-seller pair
        subset = self.df_demand[(self.df_demand['server_id'] == fs_i) & (self.df_demand['edge_id'] == es_j)]

        # if consecutive two times delay then remove the FL server from the edge server
        if len(subset) > 0:
            if list(subset['IsPred'].iloc[-1:])[0] == 1: # this FLS has pred value in the prev round
                #print(f'fs_i: {fs_i} es_j: {es_j} ############################# ############################# ############################# ############################# #############################')
                return -1.0

        #print(f'subset: {subset}, subset.shape: {subset.shape}, server_id: {fs_i}, edge_id: {es_j}')
        sequence_id = max(subset['sequence_id'])
        # Extract features and target variable
        X = subset['sequence_id'].values.reshape(-1, 1)  # Month of year as feature
        y = subset['demand'].values

        # Fit linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # prd_X = [[sequence_id+1], [sequence_id+2], [sequence_id+3]]
        prd_X = [[sequence_id + 1]]
        prd_y = model.predict(prd_X)

        # Print predicted demand for next three months
        #print(f'{fs_i}-{es_j}: {prd_y}\n')

        # if fs_i == 1:
        #     plt.title('FS_{}, ES_{}'.format(fs_i, es_j))
        #     plt.scatter(X, y, color='blue')
        #     plt.plot(X, model.predict(X.reshape(-1, 1)), color='red')
        #     plt.scatter(prd_X, prd_y, color='green')
        #     plt.show()

        return prd_y

    def get_demand_balance(self, dic_of_es_demand):
        # Step 1: Calculate the total sum of values
        total_sum = sum(dic_of_es_demand.values())

        # Step 2 and 3: Find the index that splits the values into two halves
        cumulative_sum = 0
        split_index = None

        for index, value in enumerate(dic_of_es_demand.values()):
            cumulative_sum += value
            if cumulative_sum >= total_sum / 2:
                split_index = index
                break

        # Step 4: Separate the keys into two lists based on the split index
        keys_left = list(dic_of_es_demand.keys())[:split_index]
        keys_right = list(dic_of_es_demand.keys())[split_index:]

        dic_left = {key: dic_of_es_demand[key] for key in keys_left}
        dic_right = {key: dic_of_es_demand[key] for key in keys_right}

        return keys_left, keys_left, dic_left, dic_right

    def fund_heterogeneity(self, gamma):
        min_f_0 = 0.5
        gramma_fs_list, f_i = [], []
        k = 0
        fs_list = [i for i in range(0, self.n_fs)]
        f_i = [0 for i in range(0, self.n_fs)]
        if gamma >= 0 and gamma <= 1:
            k = math.floor(gamma * self.n_fs)
            if gamma == 1:
                k = k - 1
            # gramma_fs_list = list(np.random.choice(fs_list, k, replace=False))
            for i in range(k):
                gramma_fs_list.append(fs_list[self.n_fs - (i + 1)])
        cnt = 0
        for i in fs_list:
            if i in gramma_fs_list:
                cnt = cnt + 1
                f_i[i] = min_f_0 + ((cnt * (1 - min_f_0)) / k)
            else:
                f_i[i] = min_f_0
        return f_i

    def distribute(self, u_i, frac, num_user_in_fs, f_i_list, alpha, beta, delta, total_user_each_es, threshold, rnd, fs_wait_time_sec, Rho):
        self.frac = frac
        self.num_user_in_fs = num_user_in_fs
        self.total_user_each_es = total_user_each_es
        self.u_i = self.bandwidth_Heterogeneity(u_i, delta)
        self.f_i = list(f_i_list)
        # print(f_i_list)
        fs_alpha_list, es_beta_list = [], []
        alph_fs = 0
        beta_es = 0
        if 0 < alpha <= 1:
            alph_fs = math.floor(alpha * self.n_fs)
            beta_es = math.floor(beta * self.n_es)
            fs_alpha_list = [i for i in range(0, alph_fs)]
            es_beta_list = [i for i in range(0, beta_es)]

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

        print(f'threshold: {threshold}')
        print(f'Rho: {Rho}')
        print(f'c_i_j: {self.c_i_j}')
        self.cr_i_j = copy.deepcopy(self.c_i_j)

        ### start: demand c_i_j multiply by fund
        for i in range(len(self.c_i_j)):
            temp = []
            for j in range(len(self.c_i_j[i])):
                temp.append(self.convert_to_integer(self.c_i_j[i][j] * f_i_list[i]))
            self.cr_i_j.append(temp)
        ### end

        # Convert the list to a NumPy array
        cr_i_j_array = np.array(self.cr_i_j)
        # Calculate the column-wise sum
        column_sums = np.sum(cr_i_j_array, axis=0)
        #print(f'Demand at E_j: {column_sums}')
        #print(f'rb_j:   {self.rb_j}')

        #############################################################################################################
        #############################################################################################################

        final_ar_assign_req = []
        hetero_es_list = []
        if beta_es > 0 and alph_fs > 0:
            hetero_es_list = [i for i in range(beta_es)]
        else:
            hetero_es_list.append(self.n_es)

        fs_list_at_es = []
        latency_fs_to_es = [[(fs_wait_time_sec * 0.9) for i in range(self.n_es)] for j in range(self.n_fs)]
        for es_j in range(self.n_es):
            fs_list_for_one_es_j = []
            for fs_i in range(self.n_fs):
                if self.c_i_j[fs_i][es_j] > 0:
                    fs_list_for_one_es_j.append(fs_i)
            fs_list_at_es.append(fs_list_for_one_es_j)

        es_list_at_fs = []
        for fs_i in range(self.n_fs):
            es_list_for_one_fs_i = []
            for es_j in range(self.n_es):
                if self.c_i_j[fs_i][es_j] > 0:
                    es_list_for_one_fs_i.append(es_j)
            es_list_at_fs.append(es_list_for_one_fs_i)

        # prep dataset for the first time
        for fs_i in range(self.n_fs):
            for es_j in range(self.n_es):
                if self.cr_i_j[fs_i][es_j] > 0:
                  IsRemoveJ = self.prep_es_price_dataset(fs_i, es_j, self.cr_i_j[fs_i][es_j])

        r = 0
        while (r < rnd):
            r = r + 1
            #rint(f'round: {r}')
            total_ar_assign_req = []
            price_es = []
            ar_assign_req = []
            request_sum_in_each_es = [sum(x) for x in zip(*self.cr_i_j)]  # column-wise sum

            #self.c_rb_j = copy.deepcopy(self.rb_j)

            for es_j in range(self.n_es):
                sum_cr_i_j = 0
                for fs_i in fs_list_at_es[es_j]:
                    subset = self.df_demand[(self.df_demand['server_id'] == fs_i) & (self.df_demand['edge_id'] == es_j)]

                    if latency_fs_to_es[fs_i][es_j] > fs_wait_time_sec and (r > 1 and len(subset) > 0):
                        prep_demand = self.pred_fs_demand(es_j, fs_i)
                        # if a FS server did not response within the given wait time in two consecutive time, then we remove that FS server from that edge sever
                        if prep_demand < 0:
                            if fs_i in fs_list_at_es:
                                fs_list_at_es.remove(fs_i)
                                fs_list_at_es[es_j] = fs_list_at_es
                            #print(f'->>>>> REMOVED round: {r} fs_i: {fs_i} es_j: {es_j} demand: {prep_demand}')
                            continue

                        sum_cr_i_j = sum_cr_i_j + prep_demand[0]
                        self.prep_fs_demand_dataset(fs_i, es_j, prep_demand[0], 1)  # 1 -> true, pred demand value

                    # first round; all round within waiting time and has demandelif self.cr_i_j[fs_i][es_j] > 0:
                    elif self.cr_i_j[fs_i][es_j] > 0:
                        sum_cr_i_j = sum_cr_i_j + self.cr_i_j[fs_i][es_j]
                        self.prep_fs_demand_dataset(fs_i, es_j, self.cr_i_j[fs_i][es_j], 0)  # 0 -> actual request

                price_es.append(np.round(sum_cr_i_j / self.rb_j[es_j], 3))
                #self.c_rb_j[es_j] = self.c_rb_j[es_j] - sum_cr_i_j

            # -----------------------------------------------------------------------------------------------
            # -----------------------------------------------------------------------------------------------
            # code for FL servers

            final_ar_assign_req = ar_assign_req

            max_price = max(price_es)
            min_price = min(price_es)
            diff_percentile = min_price/max_price
            #print(f'price_es: {price_es}')

            if diff_percentile > threshold:
                self.unit_price = max_price
                self.n_round = r
                #print(f'diff_percentile: {diff_percentile} max_price: {max_price}  min_price: {min_price}')
                break;

            #index_of_max_price_es = self.get_maxIndex(price_es)

            # each fs_i update it's database with received price from es_j
            take_away_request_i = []
            for fs_i in range(self.n_fs):
                take_away_request_fs_i = 0
                es_list_for_one_fs_i = es_list_at_fs[fs_i]
                for es_j in range(self.n_es):
                    if self.cr_i_j[fs_i][es_j] > 0:
                        IsRemoveJ = self.prep_es_price_dataset(fs_i, es_j, self.cr_i_j[fs_i][es_j], price_es[es_j])

                        #IsRemoveJ: True -> multiple consecutive price high, so remove the E_j from S_i and make the request 0
                        if IsRemoveJ:
                            es_list_for_one_fs_i.remove(es_j)
                            take_away_request_fs_i = take_away_request_fs_i + self.cr_i_j[fs_i][es_j]
                            self.cr_i_j[fs_i][es_j] = 0
                es_list_at_fs[fs_i] = es_list_for_one_fs_i
                take_away_request_i.append(take_away_request_fs_i)

            for fs_i in range(self.n_fs):
            #for fs_i in n_list:
                # start: remove es_j from current fs_i if that es_j assign price 0 for this fs_i
                es_list_for_one_fs_i = es_list_at_fs[fs_i]
                for es_j in es_list_for_one_fs_i:
                    if price_es[es_j] == 0:
                        es_list_for_one_fs_i.remove(es_j)
                es_list_at_fs[fs_i] = es_list_for_one_fs_i
                # end

                # ---------------------------------------------------------------------------------------
                # start: new code for Game_distributed.py
                # ---------------------------------------------------------------------------------------

                # Step 1: total demand at each es_j of all fs_i
                # S_1: [50, 40, 40, 40, 40]
                total_d = [price_es[es_j] * self.rb_j[es_j] for es_j in es_list_for_one_fs_i]

                # Step 2: find ratio of total fs_i demand at total system demand
                # total system demand / demand of fs_i in the system
                # S_1: 4.2
                p_bar = sum(total_d)/sum(self.rb_j[es_j] for es_j in es_list_for_one_fs_i)

                # Step 3: find the edge servers factors
                # S_1: [42.0, 42.0, 42.0, 42.0, 42.0]
                d_j_bar = [p_bar * self.rb_j[es_j] for es_j in es_list_for_one_fs_i]

                # Step 4: find delta d_j_bar
                # S_1: [-8.0, 2.0, 2.0, 2.0, 2.0]
                delta_d_j = [a - b for a, b in zip(d_j_bar, total_d)]

                # Step 5:ratio
                es_j_ratio = [num * Rho for num in delta_d_j]
                #print(f'fs_i: {fs_i}: es_j_ratio: {es_j_ratio}')
                # S_1: [-0.8, 0.2, 0.2, 0.2, 0.2]

                for es_j in es_list_for_one_fs_i:
                    if es_j_ratio[es_j] < 0:
                        if self.cr_i_j[fs_i][es_j] - abs(es_j_ratio[es_j]) <= 0:
                            self.cr_i_j[fs_i][es_j] = 0
                            #latency_fs_to_es[fs_i][es_j] = 999
                        else:
                            self.cr_i_j[fs_i][es_j] = round(self.cr_i_j[fs_i][es_j] - abs(es_j_ratio[es_j]), 3)
                    else:
                        self.cr_i_j[fs_i][es_j] = round(self.cr_i_j[fs_i][es_j] + abs(es_j_ratio[es_j]), 3)
                    self.prep_es_price_dataset(fs_i, es_j, self.cr_i_j[fs_i][es_j])
            #print(f'cr_i_j: {fs_i}: {self.cr_i_j}')
                # end of es_j
            # end of fs_i

        #print(f'\n After iteration')
        request_sum_in_each_es = [sum(x) for x in zip(*self.cr_i_j)]  # column-wise sum
        #print(f'request ES : {request_sum_in_each_es}')


        cr_i_j_array = np.array(self.cr_i_j)

        #print(f'cr_i_j: {self.cr_i_j}')
        # Calculate the column-wise sum
        demand_es_j = np.sum(cr_i_j_array, axis=0)
        #print(f'column-wise sum: {demand_es_j}')

        #x_i_j = [[0 for i in range(self.n_fs)] for j in range(self.n_es)]

        for fs_i in range(self.n_fs):
            for es_j in range(self.n_es):
                tt = (round((logsumexp(self.cr_i_j[fs_i][es_j]) * logsumexp(self.rb_j[es_j])) / logsumexp(demand_es_j[es_j]), 3))
                if not math.isnan(tt):
                    self.x_i_j[fs_i][es_j] = self.convert_to_integer(tt)
                else:
                    self.x_i_j[fs_i][es_j] = 0

                if self.cr_i_j[fs_i][es_j] == 0.0:
                    self.x_i_j[fs_i][es_j] = 0

                demand_es_j[es_j] = demand_es_j[es_j] - self.cr_i_j[fs_i][es_j]
                self.rb_j[es_j] = self.rb_j[es_j] - self.x_i_j[fs_i][es_j]
        #print(f'rb_j: {self.rb_j}')
        #print(f'demand_es_j:{demand_es_j}')
        #print(f'x_i_j:{self.x_i_j}')

        row_sums = [sum(row) for row in self.x_i_j]
        print(f'fs_i: {row_sums}')

        #self.df_demand.to_csv('my_dataframe.csv', index=False)
        #Read exported CSV file and print contents
        #exported_df = pd.read_csv('my_dataframe.csv')
        #print(exported_df)
        #print(self.df_price)
        print(f'n_round: {self.n_round}  unit_price: {self.unit_price}')
        unit_price = []
        unit_price.append(self.unit_price)
        
        for i in range(0, self.n_fs):
            unit_price.append(self.f_i[i] / sum(self.x_i_j[i]))

        return unit_price, self.x_i_j, self.rb_j, self.x_i_j
        #return self.n_round, unit_price, self.x_i_j, self.rb_j, self.x_i_j


        #############################################################################################################
        #############################################################################################################


if __name__ == '__main__':

    num_user_in_fs = 200
    total_user_each_es = 10
    num_servers = 5
    num_edge_server = 5
    u_i = 1
    frac = 1
    alpha_list = [0.2, 0.4, 0.6, 0.8]
    beta_list = [0.2, 0.4, 0.6, 0.8]

    # fs_list = [0.5, 0.62, 0.75, 0.88, 1.00]
    fund_list = [0.0]

    threshold = 0.9
    rnd = 100
    fs_wait_time_sec = 2
    delta = 0.0

    Rho = 0.1
    round_info = []
    for a in alpha_list:
        alpha_round_info = []
        for b in beta_list:
            for gamma in fund_list:
                obj = Game(num_servers, num_edge_server)

                f_i_list = obj.fund_heterogeneity(gamma)
                # print("\n")
                # print("~" * 100)
                # print(f'gamma: {gamma}, {f_i_list}')

                # print(f'alpha: {a}, beta: {b}')
                one_round = 0
                #one_round, u_price, xij, remain_rb_j, assigned_rr_i_j = obj.distribute(u_i, frac, num_user_in_fs, f_i_list, a, b, delta,
                #                                                            total_user_each_es, threshold, rnd, fs_wait_time_sec, Rho)

                u_price, xij, remain_rb_j, assigned_rr_i_j = obj.distribute(u_i, frac, num_user_in_fs, f_i_list, a, b, delta, total_user_each_es, threshold, rnd, fs_wait_time_sec, Rho)

                alpha_round_info.append(one_round)
                #print(f'\nAcquired clients by FS from each ES: {xij}')
                fs_ed = []
                rem_fund = []
                used_fund = []
                for i in range(len(xij)):
                    fs_ed.append(sum(xij[i]))
                #print(f'Acquired total clients by FS:        {fs_ed}')
                #print(f'remain_rb_j in each ES:              {remain_rb_j}\n')
                #print(f'u_price:              {u_price}\n')
        round_info.append(alpha_round_info)

    print(f'round_info: {round_info}')

#

