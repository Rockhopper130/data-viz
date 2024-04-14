import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from interactive import PipeLine


crops = ['RICE', 'WHEAT', 'KHARIF SORGHUM', 'RABI SORGHUM', 'SORGHUM', 'PEARL MILLET', 'MAIZE', 'FINGER MILLET', 'BARLEY', 'CHICKPEA', 'PIGEONPEA', 'MINOR PULSES', 'GROUNDNUT', 'SESAMUM', 'RAPESEED AND MUSTARD', 'SAFFLOWER', 'CASTOR', 'LINSEED', 'SUNFLOWER', 'SOYABEAN', 'OILSEEDS', 'SUGARCANE', 'COTTON', 'FRUITS', 'VEGETABLES', 'FRUITS AND VEGETABLES', 'POTATOES', 'ONION', 'FODDER']
modes = ["PRODUCTION (1000 tons)", "AREA (1000 ha)", "YIELD (Kg per ha)"]

cmap = "viridis"
border_color = "tab:black"

def main():
    st.title("Evolution of Agriculture (1966-2017)")
    
    CROP = "RICE"
    MODE = "PRODUCTION (1000 tons)"

    CROP = st.sidebar.selectbox(f"Select Crop", crops)
    MODE = st.sidebar.selectbox(f"Select Mode", modes)
    cmap = st.text_input("Select ColorMap (default : viridis)", value="viridis")
    border_color = st.text_input("Select Border Color (default : black)", value="black")

    if st.sidebar.button("Run"):
        # try:
        temp = PipeLine(CROP=CROP, mode=MODE, cmap=cmap, border_color=border_color)
        _, anim, plot = temp.main()
        st.write(anim, unsafe_allow_html=True)
        st.pyplot(plot.draw())

        # except:
            # st.write("The mode is currently not available for this crop. Please select 'AREA (1000 ha).'")

if __name__ == "__main__":
    main()
