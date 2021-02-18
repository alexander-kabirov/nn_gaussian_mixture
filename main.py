from torch import randn, rand, sigmoid, zeros, diag, ones, exp, mean, cuda, device, optim, tensor, split
from torch.nn import Module, parameter, Linear, Softmax
from torch.distributions import MultivariateNormal


class GaussianMixture(Module):
    def __init__(self, n_components, dims):  # , dropout=0.1
        super(GaussianMixture, self).__init__()
        self.mu = parameter.Parameter(randn((n_components, dims)), requires_grad=True)  # normally dist
        self.sigmasq = parameter.Parameter(rand(n_components),
                                           requires_grad=True)  # uniformly dist on the range [0,1]
        self.mixture = Linear(dims, n_components)
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        return self.softmax(sigmoid(self.mixture(x)))

    # @staticmethod
    #  def gaussian_pdf(x, mu, sigmasq):
    #    NOTE: Replaced with MultivariateNormal from torch.distributions
    #     pdf =  (1/torch.sqrt(2*np.pi*sigmasq)) * torch.exp(torch.exp((-1/(2*sigmasq)) * torch.norm((x-mu), 2, 1)**2))
    #     return pdf


def calculate_loss(mu, sigmasq, mix, x, device):
    losses = zeros(x.shape[0], device=device)
    k = mix.shape[1]
    for i in range(k):  # To Do: put k versions for method and absolute, add loss proportionally likelihood
        # likelihood_z_x = gaussian_pdf(x, mu[i,:], torch.exp(sigmasq[i]))
        dist = MultivariateNormal(mu[i, :], diag(ones(x.shape[1]).to(device)*exp(sigmasq[i])))
        likelihood_z_x = dist.log_prob(x)
        # print(likelihood_z_x.shape)
        losses += mix[:, i] * likelihood_z_x
    return mean(-losses)


if __name__ == '__main__':
    if cuda.is_available():
        device = device('cuda:0')
    else:
        device = device('cpu')
    data = tensor(df.values).float().to(device)  # import your dataset
    data_batches = split(data, 32)
    model = GaussianMixture(2, 8).to(device) # (Number of components, Number of features)
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    for epoch in range(3000):
        # normally you would go with batches here
        optimizer.zero_grad()
        mix = model(data)
        loss = calculate_loss(model.mu, model.sigmasq, mix, data, device)
        # print(loss)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        if epoch % 100 == 0:
            # print(model.mu)
            # print(model.sigmasq)
            print(loss.item())

