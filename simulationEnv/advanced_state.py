import numpy as np
from collections import deque

class AdvancedState:
    def __init__(self, embedding_dim=384, history_length=5, search_history_length=3):
        self.embedding_dim = embedding_dim
        self.history_length = history_length
        self.search_history_length = search_history_length
        
        # Primary state vector
        self.primary_state = np.zeros(embedding_dim + 1 + 1 + 1 + history_length + 5)  # Added 5 new dimensions
        
        # Secondary state dictionary
        self.secondary_state = {
            'user_history': [],
            'resource_utilization': 0.0,
            'pfc_insights': [],
            'domain_knowledge': {},
            'search_history': deque(maxlen=search_history_length),
            'user_learning_curve': [],
            'session_duration': 0,
            'breakthrough_potential': 0.0,
            'resource_allocation_history': {'FE': [], 'IE': [], 'PFC': []},
            'error_rate': []
        }
    
    def update_query_embedding(self, new_embedding):
        self.primary_state[:self.embedding_dim] = new_embedding
    
    def update_user_expertise(self, expertise):
        self.primary_state[self.embedding_dim] = expertise
    
    def update_research_progress(self, progress):
        self.primary_state[self.embedding_dim + 1] = progress
    
    def update_novelty_score(self, novelty):
        self.primary_state[self.embedding_dim + 2] = novelty
    
    def update_recent_feedback(self, feedback):
        feedback_start = self.embedding_dim + 3
        self.primary_state[feedback_start:feedback_start+self.history_length] = np.roll(
            self.primary_state[feedback_start:feedback_start+self.history_length], -1)
        self.primary_state[feedback_start+self.history_length-1] = feedback
    
    def update_shot_count(self, count):
        self.primary_state[-5] = count
    
    def update_average_shot_count(self, avg_count):
        self.primary_state[-4] = avg_count
    
    def update_shot_efficiency(self, efficiency):
        self.primary_state[-3] = efficiency
    
    def update_user_engagement(self, engagement):
        self.primary_state[-2] = engagement
    
    def update_query_complexity(self, complexity):
        self.primary_state[-1] = complexity
    
    def add_search_result(self, result):
        self.secondary_state['search_history'].append(result)
    
    def update_user_learning_curve(self, expertise):
        self.secondary_state['user_learning_curve'].append(expertise)
    
    def update_session_duration(self, duration):
        self.secondary_state['session_duration'] = duration
    
    def update_breakthrough_potential(self, potential):
        self.secondary_state['breakthrough_potential'] = potential
    
    def update_resource_allocation(self, fe, ie, pfc):
        self.secondary_state['resource_allocation_history']['FE'].append(fe)
        self.secondary_state['resource_allocation_history']['IE'].append(ie)
        self.secondary_state['resource_allocation_history']['PFC'].append(pfc)
    
    def update_error_rate(self, error_rate):
        self.secondary_state['error_rate'].append(error_rate)
    
    def get_state(self):
        return self.primary_state, self.secondary_state