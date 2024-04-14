import warnings
warnings.filterwarnings("ignore")

import os
import random
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plotnine import *

import cv2
from IPython.display import HTML
from matplotlib import animation, rc
rc('animation', html='jshtml')

import geopandas as gpd

#------------------------------------------------------------------------------------------

shp_gdf = gpd.read_file('data/india_gis/India States/Indian_states.shp')
df = pd.read_csv("data/ICRISAT-District Level Data.csv")
df = df.replace("Orissa", "Odisha")

class PipeLine():
    def __init__(self, CROP, mode, cmap, border_color):
        self.CROP = CROP
        self.mode = mode
        self.cmap = cmap
        self.bcolor = border_color
        
    def make_array(self):
        temp = []
        states = []
        for i,state in enumerate(df["State Name"].unique()):
            prod = df[df["State Name"] == state].groupby("Year")[f"{self.CROP} {self.mode}"].sum()

            if(len(prod) != 52):
                for i in range(1966,2018):
                    try:
                        _ = prod[i]
                    except:
                        prod[i] = prod.mean()
                
            states.append(state)
            temp.append(np.array(prod))
            
        temp = np.array(temp) 
        
        self.array = temp
        self.height = (self.array.max()//1000 + 1) * 1000
        self.state_list = states 
    
    def make_df(self):
        stat_df = pd.DataFrame()
        stat_df["state"] = self.state_list

        for i in range(52):
            stat_df[str(1966 + i)] = self.array[:,i]
            
        stat_df.iloc[:,1:][stat_df.iloc[:,1:] < 0] = pd.NA

        for index, row in stat_df.iterrows():
            for column in stat_df.columns:
                if pd.isna(row[column]):
                    stat_df.at[index, column] = stat_df.iloc[index][1:].mean()
                    
        self.stat_df = stat_df
                    
    def format_plot_df(self):
        plot_df = pd.merge(shp_gdf, self.stat_df, left_on='st_nm', right_on='state', how='outer')
        plot_df.drop("state", axis=1, inplace=True)
        plot_df.fillna(0, inplace=True)
        self.plot_df = plot_df
        
    def plot_map(self,title,column,name,cmap,fontsize,fontweight):
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.axis('off')
        ax.set_title(title,fontdict={'fontsize': fontsize, 'fontweight' : fontweight})
        self.plot_df.plot(column = column, cmap=self.cmap, linewidth=0.5, ax=ax, edgecolor=self.bcolor,legend=True,vmin = 0,vmax=self.height)
        plt.savefig(name)
        plt.tight_layout()
        plt.close(1)
        # plt.show()
    
    def start_plotting(self):
        cmap = self.cmap
        fontsize = '15'
        fontweight = '3'
        
        self.directory = f'plots/{self.CROP.lower()}/{self.mode.lower()}/'
        
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        for i in tqdm(range(1966,2018)):
            self.plot_map(f'{self.CROP.title()} {self.mode.title()} ({i})',f'{i}',f'{self.directory}/{i}.jpg',cmap,fontsize,fontweight)
        
    def create_animation(self,ims):
        fig=plt.figure(figsize=(10,10))
        plt.axis('off')
        im=plt.imshow(cv2.cvtColor(ims[0],cv2.COLOR_BGR2RGB))
        
        def animate_func(i):
            im.set_array(cv2.cvtColor(ims[i],cv2.COLOR_BGR2RGB))
            return [im]

        return animation.FuncAnimation(fig, animate_func, frames=len(ims), interval=500)

        
    def build_animation(self):
        imgdir1 =f'plots/{self.CROP.lower()}/{self.mode.lower()}' 
        paths0=[]
        for dirname, _, filenames in os.walk(imgdir1):
            for filename in filenames:
                if filename not in ['__notebook_source__.ipynb','__notebook__.ipynb']:
                    # print(filename)
                    paths0+=[os.path.join(dirname, filename)]     
        paths0 = sorted(paths0)
        images0=[]
        for i in tqdm(range(0,len(paths0))):
            images0+=[cv2.imread(paths0[i])]
        
        anim = self.create_animation(np.array(images0))
        plt.close(1)
        
        html_code = anim.to_html5_video()
        modified_html_code = html_code.replace('<video ', '<video width="800" height="800" ')
        return HTML(modified_html_code)
    
    def plotnine_scatter(self):
        temp = self.stat_df.iloc[:,1:].mean(axis=1).values
        top_st = self.stat_df["state"].iloc[np.argsort(temp)[::-1][:5]].values

        plt_df = self.stat_df[self.stat_df["state"].isin(top_st)]
        plt_df = plt_df.melt(id_vars='state')
        plt_df['value'] = plt_df['value'].astype(float)
        plt_df['variable'] = plt_df['variable'].astype(int)

        plot = (
            ggplot(plt_df, aes(x='variable', y='value', color='state')) +
            geom_line() +
            geom_point() +
            labs(x="Year", y=self.mode, title=f"{self.mode} (1966-2017) - ({self.CROP.title()})") +
            theme(axis_text_x=element_text(angle=45, hjust=1)) +
            scale_color_brewer(type='qual', palette='Set1') +
            theme(figure_size=(15, 6)) +
            scale_y_continuous(breaks=range(0, int(plt_df['value'].max()) + 1000, int(self.height//10))) +
            scale_x_continuous(breaks=range(1966, 2018))
        )
        
        return plot
    
    
    def main(self):
        self.make_array()
        self.make_df()
        self.format_plot_df()
        
        self.directory = f'plots/{self.CROP.lower()}/{self.mode.lower()}'
        if not os.path.exists(self.directory):
            self.start_plotting()
        
        anim = self.build_animation()
        plot = self.plotnine_scatter()
        
        return self.plot_df, anim, plot