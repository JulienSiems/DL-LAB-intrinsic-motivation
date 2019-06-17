import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class IntrinsicRewardGenerator:
    def __init__(self, state_encoder, inverse_dynamics_model, forward_dynamics_model, num_actions):
        self.state_encoder = state_encoder
        self.inverse_dynamics_model = inverse_dynamics_model
        self.forward_dynamics_model = forward_dynamics_model
        self.num_actions = num_actions

    def compute_intrinsic_reward(self, state, action, next_state):
        phi_s_t = self.state_encoder(torch.from_numpy(state).to(device))  # [BS, 288]
        phi_s_tp1 = self.state_encoder(torch.from_numpy(next_state).to(device))  # [BS, 288]

        action_pred = self.inverse_dynamics_model(phi_s_t, phi_s_tp1)  # [BS, n_a]

        action = torch.from_numpy(action).to(device)
        action_one_hot = torch.nn.functional.one_hot(action.long(), num_classes=self.num_actions)
        phi_s_tp1_pred = self.forward_dynamics_model(phi_s_t, action_one_hot.float())

        L_I = torch.nn.CrossEntropyLoss()(action_pred, action.long())  # (eq. 3)
        r_i = 1 / 2 * ((phi_s_tp1 - phi_s_tp1_pred) ** 2).sum(dim=1)  # (eq. 6)
        L_F = r_i.mean()  # (eq. 5)
        return L_I, L_F, r_i
