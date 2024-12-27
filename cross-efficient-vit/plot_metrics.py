import matplotlib.pyplot as plt

def save_plots(train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list, 
               train_precision_list, val_precision_list, num_epochs):
    # Plot and save Training and Validation Loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_loss_list, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_plot.png')

    # Plot and save Training and Validation Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_accuracy_list, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracy_list, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig('accuracy_plot.png')

    # Plot and save Training and Validation Precision
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_precision_list, label='Train Precision')
    plt.plot(range(1, num_epochs + 1), val_precision_list, label='Validation Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Training and Validation Precision')
    plt.legend()
    plt.savefig('precision_plot.png')
