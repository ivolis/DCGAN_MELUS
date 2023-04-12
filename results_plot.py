import matplotlib.pyplot as plt

import pandas as pd

# Load CSV file into a pandas DataFrame
df = pd.read_csv("./results.csv")

# Access columns by name
epochs = df["epoch"]
discriminator_loss_real = df["discriminator_loss_real"]
discriminator_loss_fake = df["discriminator_loss_fake"]
generator_loss = df["generator_loss"]
discriminator_acc_real = df["discriminator_acc_real"] 
discriminator_acc_fake = df["discriminator_acc_fake"]

# Plot and save
plt.plot(epochs, discriminator_loss_real)
plt.plot(epochs, discriminator_loss_fake)
plt.plot(epochs, generator_loss)
plt.legend(['Discriminator loss (real imgs)', 'Discriminator loss (fake imgs)', 'Generator loss'])
plt.grid()
plt.savefig('loss.png', bbox_inches='tight')
plt.close()

plt.plot(epochs, discriminator_acc_real)
plt.plot(epochs, discriminator_acc_fake)
plt.legend(['Discriminator accuracy (real imgs)', 'Discriminator accuracy (fake imgs)'])
plt.grid()
plt.savefig('acc.png', bbox_inches='tight')
plt.close()