import torch
import torchaudio
import matplotlib.pyplot as plt
from tqdm import tqdm
from nablafx.processors import StaticMLPNonlinearity, StaticFIRFilter

"""
Pretrain a StaticMLPNonlinearity or StaticFIRFilter blocks
"""

if __name__ == "__main__":
    mode = "firfilter"

    sample_rate = 48000
    n_iter = 5000

    if mode == "nonlinearity":
        hidden_dim = 64
        num_layers = 3
        w0_initial = 30.0
        pretrained = False

        # Create a MLPNonlinearity object
        mlp_nonlinearity = StaticMLPNonlinearity(
            sample_rate,
            hidden_dim,
            num_layers,
            w0_initial,
            pretrained,
        )

        optimizer = torch.optim.Adam(mlp_nonlinearity.parameters(), lr=1e-3)

        # create a tensor of coordinates
        x = torch.linspace(-4, 4, 100)
        x = x.view(1, 1, -1)

        # create the target
        y = torch.tanh(x)

        control_params = None
        train = True
        y_hat, _ = mlp_nonlinearity(x, control_params, False)

        pbar = tqdm(range(n_iter))

        for n in pbar:
            optimizer.zero_grad()
            y_hat, _ = mlp_nonlinearity(x, control_params, train)
            loss = torch.nn.functional.mse_loss(y_hat, y)
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss: {loss.item():.4e}")

        y_hat = y_hat.detach()

        mlp_nonlinearity.eval()

        with torch.no_grad():
            y_hat, _ = mlp_nonlinearity(x, control_params)

        plt.plot(x.squeeze(), y.squeeze(), label="target")
        plt.plot(x.squeeze(), y_hat.squeeze(), label="prediction")
        plt.legend()
        plt.grid(c="lightgray")
        plt.savefig(f"weights/static_mlp_nonlinearity_h{hidden_dim}-l{num_layers}_tanh.png", dpi=300)

        torch.save(
            mlp_nonlinearity.state_dict(),
            f"weights/static_mlp_nonlinearity_h{hidden_dim}-l{num_layers}_tanh.pt",
        )

    elif mode == "firfilter":
        n_taps = 64
        hidden_dim = 32
        num_layers = 5
        w0_initial = 1000.0
        pretrained = None

        # load IR
        y, sr = torchaudio.load("weights/Fredman Straight.wav")

        # resample to target sample rate
        if sr != sample_rate:
            y = torchaudio.functional.resample(y, sr, sample_rate)

        # take left channel only if stereo
        if y.shape[0] > 1:
            y = y[0:1, :]

        # truncate to target samples
        if y.shape[-1] >= n_taps:
            y = y[:, :n_taps]
        else:  # or zero pad to the target samples
            y = torch.nn.functional.pad(y, (0, n_taps - y.shape[-1]))

        # normalize
        y /= (y**2).sum()

        # create a FIR object
        firfilter = StaticFIRFilter(
            sample_rate,
            n_taps,
            hidden_dim,
            num_layers,
            w0_initial,
            pretrained,
        )
        firfilter.cuda()

        optimizer = torch.optim.Adam(firfilter.parameters(), lr=1e-3)

        # create a tensor of coordinates
        x = torch.linspace(-1, 1, n_taps)
        x = x.view(1, 1, -1)
        x = x.cuda()

        # create the target
        y = y.type_as(x)

        control_params = None
        train = True
        y_hat, _ = firfilter(x, control_params, False)

        pbar = tqdm(range(n_iter))

        for n in pbar:
            optimizer.zero_grad()
            y_hat = firfilter.extract_impulse_response(x)
            loss = torch.nn.functional.mse_loss(y_hat.squeeze(), y.squeeze())
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss: {loss.item():.4e}")

        y = y.cpu()
        y_hat = y_hat.detach().cpu()
        x = x.cpu()
        firfilter.cpu()
        firfilter.eval()

        with torch.no_grad():
            y_hat = firfilter.extract_impulse_response(x)

        fig, axs = plt.subplots(nrows=2, ncols=1)
        axs[0].plot(x.squeeze(), y.squeeze(), label="target")
        axs[0].plot(x.squeeze(), y_hat.squeeze(), label="prediction")
        axs[0].legend()
        axs[0].grid(c="lightgray")
        axs[0].set_xlim(x.squeeze()[0], x.squeeze()[-1])

        axs[1].plot(x.squeeze(), y.squeeze(), label="target")
        axs[1].plot(x.squeeze(), y_hat.squeeze(), label="prediction")
        axs[1].legend()
        axs[1].grid(c="lightgray")
        axs[1].set_xlim(x.squeeze()[0], x.squeeze()[n_taps // 4])
        plt.savefig(f"weights/static_fir_filter_t{n_taps}-h{hidden_dim}-l{num_layers}.png", dpi=300)

        torch.save(
            firfilter.state_dict(),
            f"weights/static_fir_filter_t{n_taps}-h{hidden_dim}-l{num_layers}.pt",
        )
