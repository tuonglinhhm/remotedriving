import math
import random

import numpy as np


class UAVEnv(object):
    height = ground_length = ground_width = 100 
    sum_task_size = 100 * 1048576  
    loc_uav = [50, 50]
    bandwidth_nums = 1
    B = bandwidth_nums * 10 ** 6 
    p_noisy_los = 10 ** (-13)  
    p_noisy_nlos = 10 ** (-11) 
    flight_speed = 50.  
    # f_ue = 6e8  
    f_ue = 6e8  
    f_uav = 1.2e9  
    r = 10 ** (-27) 
    s = 1000 
    p_uplink = 0.1  
    alpha0 = 0.001 
    T = 320 
    t_fly = 1
    t_com = 7
    delta_t = t_fly + t_com  
    v_ue = 1    
    slot_num = int(T / delta_t)  
    m_uav = 9.65  
    e_battery_uav = 500000 

    #################### ues ####################
    M = 4  
    block_flag_list = np.random.randint(0, 2, M)  
    loc_ue_list = np.random.randint(0, 101, size=[M, 2])  
    # task_list = np.random.randint(1572864, 2097153, M)     
    task_list = np.random.randint(2097153, 2621440, M)  #
    action_bound = [-1, 1]  
    action_dim = 4  
    state_dim = 4 + M * 4 

    def __init__(self):

        self.start_state = np.append(self.e_battery_uav, self.loc_uav)
        self.start_state = np.append(self.start_state, self.sum_task_size)
        self.start_state = np.append(self.start_state, np.ravel(self.loc_ue_list))
        self.start_state = np.append(self.start_state, self.task_list)
        self.start_state = np.append(self.start_state, self.block_flag_list)
        self.state = self.start_state

    def reset_env(self):
        self.sum_task_size = 100 * 1048576 
        self.e_battery_uav = 500000 
        self.loc_uav = [50, 50]
        self.loc_ue_list = np.random.randint(0, 101, size=[self.M, 2]) 
        self.reset_step()

    def reset_step(self):
      
        self.task_list = np.random.randint(2621440, 3145729, self.M)  
        
        self.block_flag_list = np.random.randint(0, 2, self.M)  

    def reset(self):
        self.reset_env()

        self.state = np.append(self.e_battery_uav, self.loc_uav)
        self.state = np.append(self.state, self.sum_task_size)
        self.state = np.append(self.state, np.ravel(self.loc_ue_list))
        self.state = np.append(self.state, self.task_list)
        self.state = np.append(self.state, self.block_flag_list)
        return self._get_obs()

    def _get_obs(self):
      
        self.state = np.append(self.e_battery_uav, self.loc_uav)
        self.state = np.append(self.state, self.sum_task_size)
        self.state = np.append(self.state, np.ravel(self.loc_ue_list))
        self.state = np.append(self.state, self.task_list)
        self.state = np.append(self.state, self.block_flag_list)
        return self.state

    def step(self, action):  
        step_redo = False
        is_terminal = False
        remote_mission_ratio_change = False
        reset_dist = False
        action = (action + 1) / 2  
        if action[0] == 1:
            ue_id = self.M - 1
        else:
            ue_id = int(self.M * action[0])

        theta = action[1] * np.pi * 2  
        remote_mission_ratio = action[3]
        task_size = self.task_list[ue_id]
        block_flag = self.block_flag_list[ue_id]

       
        dis_fly = action[2] * self.flight_speed * self.t_fly  
       
        e_fly = (dis_fly / self.t_fly) ** 2 * self.m_uav * self.t_fly * 0.5 

       
        dx_uav = dis_fly * math.cos(theta)
        dy_uav = dis_fly * math.sin(theta)
        loc_uav_after_fly_x = self.loc_uav[0] + dx_uav
        loc_uav_after_fly_y = self.loc_uav[1] + dy_uav

       
        t_server = remote_mission_ratio * task_size / (self.f_uav / self.s)
        e_server = self.r * self.f_uav ** 3 * t_server  

        if self.sum_task_size == 0: 
            is_terminal = True
            reward = 0
        elif self.sum_task_size - self.task_list[ue_id] < 0: 
            self.task_list = np.ones(self.M) * self.sum_task_size
            reward = 0
            step_redo = True
        elif loc_uav_after_fly_x < 0 or loc_uav_after_fly_x > self.ground_width or loc_uav_after_fly_y < 0 or loc_uav_after_fly_y > self.ground_length:  # uav位置不对
            
            reset_dist = True
            delay = self.com_delay(self.loc_ue_list[ue_id], self.loc_uav, remote_mission_ratio, task_size, block_flag)  # 计算delay
            reward = -delay
           
            self.e_battery_uav = self.e_battery_uav - e_server  
            self.reset2(delay, self.loc_uav[0], self.loc_uav[1], remote_mission_ratio, task_size, ue_id)
        elif self.e_battery_uav < e_fly or self.e_battery_uav - e_fly < e_server:  
            delay = self.com_delay(self.loc_ue_list[ue_id], np.array([loc_uav_after_fly_x, loc_uav_after_fly_y]),
                                   0, task_size, block_flag) 
            reward = -delay
            self.reset2(delay, loc_uav_after_fly_x, loc_uav_after_fly_y, 0, task_size, ue_id)
            remote_mission_ratio_change = True
        else: 
            delay = self.com_delay(self.loc_ue_list[ue_id], np.array([loc_uav_after_fly_x, loc_uav_after_fly_y]),
                                   remote_mission_ratio, task_size, block_flag)
            reward = -delay
            
            self.e_battery_uav = self.e_battery_uav - e_fly - e_server 
            self.loc_uav[0] = loc_uav_after_fly_x  
            self.loc_uav[1] = loc_uav_after_fly_y
            self.reset2(delay, loc_uav_after_fly_x, loc_uav_after_fly_y, remote_mission_ratio, task_size,
                                           ue_id)   

        return self._get_obs(), reward, is_terminal, step_redo, remote_mission_ratio_change, reset_dist

   
    def reset2(self, delay, x, y, remote_mission_ratio, task_size, ue_id):
        self.sum_task_size -= self.task_list[ue_id]  
        for i in range(self.M):  
            tmp = np.random.rand(2)
            theta_ue = tmp[0] * np.pi * 2  
            dis_ue = tmp[1] * self.delta_t * self.v_ue  
            self.loc_ue_list[i][0] = self.loc_ue_list[i][0] + math.cos(theta_ue) * dis_ue
            self.loc_ue_list[i][1] = self.loc_ue_list[i][1] + math.sin(theta_ue) * dis_ue
            self.loc_ue_list[i] = np.clip(self.loc_ue_list[i], 0, self.ground_width)
        self.reset_step() 
       
        file_name = 'output.txt'
     
        with open(file_name, 'a') as file_obj:
            file_obj.write("\nUE-" + '{:d}'.format(ue_id) + ", task size: " + '{:d}'.format(int(task_size)) + ", remote-mission ratio:" + '{:.2f}'.format(remote_mission_ratio))
            file_obj.write("\ndelay:" + '{:.2f}'.format(delay))
            file_obj.write("\nUAV hover loc:" + "[" + '{:.2f}'.format(x) + ', ' + '{:.2f}'.format(y) + ']') 


    
    def com_delay(self, loc_ue, loc_uav, remote_mission_ratio, task_size, block_flag):
        dx = loc_uav[0] - loc_ue[0]
        dy = loc_uav[1] - loc_ue[1]
        dh = self.height
        dist_uav_ue = np.sqrt(dx * dx + dy * dy + dh * dh)
        p_noise = self.p_noisy_los
        if block_flag == 1:
            p_noise = self.p_noisy_nlos
        g_uav_ue = abs(self.alpha0 / dist_uav_ue ** 2)  
        trans_rate = self.B * math.log2(1 + self.p_uplink * g_uav_ue / p_noise)  
        t_tr = remote_mission_ratio * task_size / trans_rate
        t_edge_com = remote_mission_ratio * task_size / (self.f_uav / self.s)
        t_local_com = (1 - remote_mission_ratio) * task_size / (self.f_ue / self.s)
        if t_tr < 0 or t_edge_com < 0 or t_local_com < 0:
            raise Exception(print("+++++++++++++++++!! error !!+++++++++++++++++++++++"))
        return max([t_tr + t_edge_com, t_local_com])  
