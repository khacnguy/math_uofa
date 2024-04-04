from cnn import mnist_train, mnist_fashion_train
from svm import mnist_train_svm
import matplotlib.pyplot as plt
import pickle
from numpy import mean
def summarize_diagnostics(histories):
		# plot loss
    for history in histories:
      plt.subplot(211)
      #plt.title('Cross Entropy Loss')
      plt.plot(history.history['loss'], color='blue', label='train')
      plt.plot(history.history['val_loss'], color='orange', label='test')
      plt.subplot(211).set(ylabel = "Cross entropy loss")
      # plot accuracy
      plt.subplot(212)
      #plt.title('Classification Accuracy')
      plt.plot(history.history['accuracy'], color='blue', label='train')
      plt.plot(history.history['val_accuracy'], color='orange', label='test')
      plt.subplot(212).set(xlabel = 'Epoch', ylabel = 'Accuracy on validation set')
    plt.show()

# summarize model performance
def summarize_performance(score):
	# print summary
	print('Accuracy: %.3f ' % mean(score))

if __name__=="__main__":
    histories, accuracies = mnist_train()
    summarize_diagnostics(histories)
    summarize_performance(accuracies)
    histories, accuracies = mnist_fashion_train()
    summarize_diagnostics(histories)
    summarize_performance(accuracies)
    mnist_train_svm(low_res = True)
    mnist_train_svm(low_res = False)

