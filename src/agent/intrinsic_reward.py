import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class IntrinsicRewardGenerator:
    def __init__(self, state_encoder, inverse_dynamics_model, forward_dynamics_model, num_actions, fixed_encoder, residual_forward, use_history):
        self.state_encoder = state_encoder
        self.inverse_dynamics_model = inverse_dynamics_model
        self.forward_dynamics_model = forward_dynamics_model
        self.num_actions = num_actions
        self.fixed_encoder = fixed_encoder
        self.residual = residual_forward
        self.use_history = use_history

    def compute_intrinsic_reward(self, state, action, next_state):
        if not self.use_history:
            state = np.expand_dims(state[:, -1, :, :],axis=1)
            next_state = np.expand_dims(next_state[:, -1, :, :],axis=1)

        phi_s_t = self.state_encoder(torch.from_numpy(state).to(device).float())  # [BS, 288]
        phi_s_tp1 = self.state_encoder(torch.from_numpy(next_state).to(device).float())  # [BS, 288]
        if self.fixed_encoder:
            phi_s_t = phi_s_t.detach()
            phi_s_tp1 = phi_s_tp1.detach()

        action_pred = self.inverse_dynamics_model(phi_s_t, phi_s_tp1)  # [BS, n_a]

        action = torch.from_numpy(action).to(device)
        action_one_hot = torch.nn.functional.one_hot(action.long(), num_classes=self.num_actions)
        if self.residual:
            phi_s_tp1_pred = phi_s_t + self.forward_dynamics_model(phi_s_t, action_one_hot.float())
        else:
            phi_s_tp1_pred = self.forward_dynamics_model(phi_s_t, action_one_hot.float())

        l_i = torch.nn.CrossEntropyLoss(reduction='none')(action_pred, action.long())  # (eq. 3)
        L_I = l_i.mean()
        r_i = 0.5 * ((phi_s_tp1 - phi_s_tp1_pred) ** 2).sum(dim=1)  # (eq. 6)
        L_F = r_i.mean()  # (eq. 5)
        return L_I, L_F, r_i, l_i
