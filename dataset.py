import numpy as np
import os
import torch


class Walker2dImitationData:
    def __init__(self, seq_len):
        self.seq_len = seq_len
        all_files = sorted(
            [
                os.path.join("data/walker", d)
                for d in os.listdir("data/walker")
                if d.endswith(".npy")
            ]
        )

        self.rng = np.random.RandomState(891374)
        np.random.RandomState(125487).shuffle(all_files)
        # 15% test set, 10% validation set, the rest is for training
        test_n = int(0.15 * len(all_files))
        valid_n = int((0.15 + 0.1) * len(all_files))
        test_files = all_files[:test_n]
        valid_files = all_files[test_n:valid_n]
        train_files = all_files[valid_n:]

        train_x, train_t, train_y = self._load_files(train_files)
        valid_x, valid_t, valid_y = self._load_files(valid_files)
        test_x, test_t, test_y = self._load_files(test_files)

        train_x, train_t, train_y = self.perturb_sequences(train_x, train_t, train_y)
        valid_x, valid_t, valid_y = self.perturb_sequences(valid_x, valid_t, valid_y)
        test_x, test_t, test_y = self.perturb_sequences(test_x, test_t, test_y)

        self.train_x, self.train_times, self.train_y = self.align_sequences(
            train_x, train_t, train_y
        )
        self.valid_x, self.valid_times, self.valid_y = self.align_sequences(
            valid_x, valid_t, valid_y
        )
        self.test_x, self.test_times, self.test_y = self.align_sequences(
            test_x, test_t, test_y
        )
        self.input_size = self.train_x.shape[-1]

        print("train_times: ", str(self.train_times.shape))
        print("train_x: ", str(self.train_x.shape))
        print("train_y: ", str(self.train_y.shape))

    def align_sequences(self, set_x, set_t, set_y):

        times = []
        x = []
        y = []
        for i in range(len(set_y)):

            seq_x = set_x[i]
            seq_t = set_t[i]
            seq_y = set_y[i]

            for t in range(0, seq_y.shape[0] - self.seq_len, self.seq_len // 4):
                x.append(seq_x[t: t + self.seq_len])
                times.append(seq_t[t: t + self.seq_len + 1])
                y.append(seq_y[t: t + self.seq_len])

        return (
            np.stack(x, axis=0),
            np.expand_dims(np.stack(times, axis=0), axis=-1),
            np.stack(y, axis=0),
        )

    def perturb_sequences(self, set_x, set_t, set_y):

        x = []
        times = []
        y = []
        for i in range(len(set_y)):

            seq_x = set_x[i]
            seq_y = set_y[i]

            new_x, new_times = [], []
            new_y = []

            skip = 0
            t = 0
            new_times.append(t)
            while t < seq_y.shape[0]:
                new_x.append(seq_x[t])
                new_y.append(seq_y[t])
                skip = self.rng.randint(1, 5)
                t += skip
                new_times.append(t)
            x.append(np.stack(new_x, axis=0))
            times.append(np.stack(new_times, axis=0))
            y.append(np.stack(new_y, axis=0))
        return x, times, y

    def _load_files(self, files):
        all_x = []
        all_t = []
        all_y = []
        for f in files:

            arr = np.load(f)
            x_state = arr[:-1, :].astype(np.float32)    # (s_0, s_62)
            y = arr[1:, :].astype(np.float32)           # (s_1, s_63)

            x_times = np.ones(x_state.shape[0])
            all_x.append(x_state)
            all_t.append(x_times)
            all_y.append(y)

            # print("Loaded file '{}' of length {:d}".format(f, x_state.shape[0]))
        return all_x, all_t, all_y


class WalkerTorchDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.t = t

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.t[i]


class MujocoDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.datadim = self.data.shape[-1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x: np.array = self.data[i]
        y: np.array = np.concatenate([x[1:], x[-1][np.newaxis, :]])
        return x, y


if __name__ == '__main__':
    data = Walker2dImitationData(seq_len=64)
    print(data.train_x.shape, data.train_times.shape, data.train_y.shape)
