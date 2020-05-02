from interface import implements
from IModel import IModel
import tensorflow as tf
from tensorflow import losses 
from PerformanceCounter import PerformanceCounter

class TensorflowModel(implements(IModel)):

    def __init__(self, model): 
        self.Model = model 
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.PerformanceCounter = PerformanceCounter() 

    def Prepare(self):
        pass

    def Loss(self, model, x, y, training):
        y_ = self.Model(x, training=training)
        return self.loss_object(y_true=y, y_pred=y_)

    def grad(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.Loss(model, inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def Fit(self, generator): 
        
        usageStates = [] 
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.BinaryAccuracy() 

        for epoch in range(5):

            print("processing epoch:", epoch, "of:", )

            batches = len(generator)

            self.PerformanceCounter.Start() 

            for batch in range(batches): 

                print("processing batch:", batch, "of:", batches, "Elapsed time:", self.PerformanceCounter.Waypoint(), epoch_accuracy.result())

                features, labels = generator.__getitem__(batch)
                loss_value, grads = self.grad(self.Model, features, labels)
                optimizer.apply_gradients(zip(grads, self.Model.trainable_variables))

                epoch_loss_avg.update_state(loss_value)
                epoch_accuracy.update_state(labels, self.Model(features, training=True)) 
            
            print("epoch loss:", epoch_loss_avg.result(), "Epoch accuracy:", epoch_accuracy.result(), "Total time:", self.PerformanceCounter.Stop())

            usageStates.append(self.PerformanceCounter.GetUsageStats()) 
            epoch_loss_avg.reset_states()
            epoch_accuracy.reset_states()

        return usageStates

    def Test(self, generator):
        
        batches = len(generator)
        test_accuracy = tf.keras.metrics.BinaryAccuracy() 

        for batch in range(batches): 

                features, labels = generator.__getitem__(batch)
                test_accuracy.update_state(labels, self.Model(features, training=False)) 

        print ("Total test accuracy:", test_accuracy.result())
        return test_accuracy.result()