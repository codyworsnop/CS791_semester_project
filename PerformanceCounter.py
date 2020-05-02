import time
import psutil
import os 
import nvidia_smi

class PerformanceCounter(): 

    def __init__(self):

        nvidia_smi.nvmlInit()
        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    def Start(self): 

        self.MemoryUsage = [] 
        self.CpuUsage = [] 
        self.ElapsedTime = [] 
        self.gpu_usage = []
        self.gpu_mem = []

        self.StartTime = time.perf_counter() 
        self.WaypointStartTime = self.StartTime

    def Waypoint(self): 

        elapsedTime = time.perf_counter() - self.WaypointStartTime
        self.WaypointStartTime = time.perf_counter() 

        memoryUsage = psutil.virtual_memory()[2]
        cpuUsage = psutil.cpu_percent()
        
        self.MemoryUsage.append(memoryUsage)
        self.CpuUsage.append(cpuUsage)
        self.ElapsedTime.append(elapsedTime)

        mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
        res = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)

        self.gpu_mem.append(mem_res.used / mem_res.total)
        self.gpu_usage.append(res.gpu) 

        return elapsedTime, cpuUsage, memoryUsage, mem_res.used / mem_res.total, res.gpu

    def Stop(self):
        totalTime = time.perf_counter()  - self.StartTime
        mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)

        self.ElapsedTime.append(totalTime) 

        return totalTime, psutil.cpu_percent(), psutil.virtual_memory()[2], mem_res.used / mem_res.total, mem_res.used / mem_res.total

    def GetUsageStats(self): 

        return self.CpuUsage, self.MemoryUsage, self.ElapsedTime, self.gpu_mem, self.gpu_usage
