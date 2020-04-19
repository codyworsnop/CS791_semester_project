import time
import psutil
import os 

class PerformanceCounter(): 

    def __init__(self): 

        self.MemoryUsage = [] 
        self.CpuUsage = [] 

    def Start(self): 

        self.StartTime = time.perf_counter() 
        self.WaypointStartTime = self.StartTime

    def Waypoint(self): 

        elapsedTime = time.perf_counter() - self.WaypointStartTime
        self.WaypointStartTime = time.perf_counter() 

        memoryUsage = psutil.virtual_memory()[2]
        cpuUsage = psutil.cpu_percent()
        
        self.MemoryUsage.append(memoryUsage)
        self.CpuUsage.append(cpuUsage)

        return elapsedTime, cpuUsage, memoryUsage

    def Stop(self):
        return time.perf_counter()  - self.StartTime, psutil.cpu_percent(), psutil.virtual_memory()[2]

    def GetUsageStats(self): 

        return self.CpuUsage, self.MemoryUsage 
