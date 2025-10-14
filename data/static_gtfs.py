import partridge as ptg
import matplotlib.pyplot as plt
import pandas as pd



class StaticGTFS:
    
    def __init__(self,gtfs_path='data/budapest_gtfs.zip'):
        feed = ptg.load_feed(gtfs_path, view=None)
        self.routes = feed.routes
        self.stops = feed.stops
        self.routes = feed.trips
        self.stop_times = feed.stop_times
        
    def plot_stops(self,ax):
        ax.scatter(self.stops.stop_lon, self.stops.stop_lat, s=10, c='red', alpha=0.7)

    def plot_shapes(shapes,ax,c='black',alpha=1,lw=0.5):
        for shape_id, group in shapes.groupby("shape_id"):
            group = group.sort_values("shape_pt_sequence")
            ax.plot(group["shape_pt_lon"], group["shape_pt_lat"], linewidth=lw,c=c,alpha=alpha)
    
    
if __name__ == '__main__':
    print(StaticGTFS().routes)        