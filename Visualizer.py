import matplotlib.pyplot as plt

class Visualizer():

    def Graph(self, data): 
        
        CpuUsage_avg = [] 
        MemoryUsage_avg = [] 
        ElapsedTime_avg = [] 
        gpu_mem_avg = [] 
        gpu_usage_avg = [] 
        for epoch in data: 

            CpuUsage_avg.append(sum(epoch[0]) / len(epoch[0]))
            MemoryUsage_avg.append(sum(epoch[1]) / len(epoch[1]))
            ElapsedTime_avg.append(sum(epoch[2]) / len(epoch[2]))
            gpu_mem_avg.append(sum(epoch[3]) / len(epoch[3]))
            gpu_usage_avg.append(sum(epoch[4]) / len(epoch[4]))


        #Plot GPU and CPU usage 
        plt.plot(CpuUsage_avg, marker='D') 
        plt.plot(gpu_usage_avg, marker='D') 
        plt.legend(['CPU Usage %', 'GPU Usage %'])
        plt.xlabel('Epoch')
        plt.ylabel('Usage')
        plt.title('Average Usage Percent per epoch')
        plt.xticks([0, 1, 2, 3, 4])
        plt.show()

        #Plot Memory usage
        plt.plot(MemoryUsage_avg, marker='D') 
        plt.plot(gpu_mem_avg, marker='D') 
        plt.legend(['System Memory %', 'GPU Memory %'])
        plt.xlabel('Epoch')
        plt.ylabel('Memory Usage')
        plt.title('Average Memory Usage Percent per epoch')
        plt.xticks([0, 1, 2, 3, 4])
        plt.show()

        #Plot time
        plt.plot(ElapsedTime_avg, marker='D') 
        plt.legend(['Elapsed Time'])
        plt.xlabel('Epoch')
        plt.ylabel('Seconds')
        plt.title('Average training time per epoch')
        plt.xticks([0, 1, 2, 3, 4])
        plt.show()