import random

from camel import array as ca
from camel.data import DataLoader
from camel.ops import Ops, Vbuf
from camel.tensor import Tensor
from camel.nn import CNN
from camel.optim import Adam


# example training loop: classify noisy 8x8 bars with a small CNN
BACKEND = "naive" # change to "cuda" for the NVIDIA backend
EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.01


def make_bars(count, seed):
    rng = random.Random(seed)
    pixels, labels = [], []
    for i in range(count):
        label = i % 2 # balanced classes: 0 = horizontal, 1 = vertical
        position = rng.randint(1, 5)
        brightness = rng.uniform(0.8, 1.2)
        for row in range(8):
            for col in range(8):
                coordinate = row if label == 0 else col
                bar = brightness if position <= coordinate < position + 2 else 0.0
                pixels.append(bar + rng.gauss(0.0, 0.1))
        labels.append([1, 0] if label == 0 else [0, 1])
    return ca.CamelArray(pixels).reshape(count, 1, 8, 8), ca.CamelArray(labels)


def evaluate(model, images, labels):
    x = Tensor(Vbuf(images), requires_grad=False)
    y = Tensor(Vbuf(labels), requires_grad=False)
    logits = model(x)
    loss = logits.softmax_xent(y).buf.data.item()
    scores = logits.buf.data
    # larger logit is the predicted class, no softmax needed for accuracy
    correct = sum(int(scores[i, 1] > scores[i, 0]) == int(labels[i, 1])
                  for i in range(images.shape[0]))
    return loss, correct / images.shape[0]


def main():
    Ops.set_backend(BACKEND)
    ca.random.seed(0) # model initialisation
    random.seed(0) # DataLoader shuffling
    train_x, train_y = make_bars(32, seed=1)
    val_x, val_y = make_bars(16, seed=2) # fresh noise and bar positions
    loader = DataLoader(train_x, train_y, BATCH_SIZE, shuffle=True)
    model = CNN(1, [4], 2, image_size=8)
    opt = Adam(model.parameters(), lr=LEARNING_RATE)

    val_loss, accuracy = evaluate(model, val_x, val_y)
    print(f"backend: {BACKEND}, train: 32 images, validation: 16 images")
    print(f"before training: val loss: {val_loss:.4f}, val accuracy: {accuracy:.1%}")

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for images, labels in loader:
            x = Tensor(Vbuf(images), requires_grad=False)
            y = Tensor(Vbuf(labels), requires_grad=False)
            opt.zero_grad()
            loss = model(x).softmax_xent(y)
            total_loss += loss.buf.data.item() * images.shape[0]
            loss.backward()
            opt.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            # validation: forward only, no parameter updates
            val_loss, accuracy = evaluate(model, val_x, val_y)
            print(f"epoch: {epoch:2d}, train loss: {total_loss / train_x.shape[0]:.4f}, "
                  f"val loss: {val_loss:.4f}, val accuracy: {accuracy:.1%}")


if __name__ == "__main__":
    main()
