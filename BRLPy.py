import tkinter as tk
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from itertools import count, cycle
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import numpy as np
import scipy.special as special
from scipy.integrate import quad
import pandas as pd
from scipy.stats import linregress

###########################################################################################

#Russell Burns

###########################################################################################

m_i = 6.7e-26
#AN=18 #atomic number
e = 1.60217663 * 10**-19
m_e = 9.1093837e-31



###########################################################################################

class ImageLabel(tk.Label): #from my kaleido repository
    """
    A Label that displays images, and plays them if they are gifs
    :im: A PIL Image instance or a string filename
    """
    def load(self, im):
        if isinstance(im, str):
            im = Image.open(im)
        frames = []

        try:
            for i in count(1):
                frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(i)
        except EOFError:
            pass
        self.frames = cycle(frames)

        try:
            self.delay = im.info['duration']
        except:
            self.delay = 100

        self.next_frame()

    def next_frame(self):
        if self.frames:
            self.config(image=next(self.frames))
            self.after(self.delay, self.next_frame)

###########################################################################################

root = tk.Tk()
root.title("IV Curve GUI")

lbl = ImageLabel(root)
lbl.pack()
lbl.load('starsmall.gif')

###########################################################################################
voltage_frame = LabelFrame(root, text = "Voltage Range",bg="#ccef1b", font="Z003")
voltage_frame.pack()
voltage_frame.place(relx=.1,rely=.9,anchor=CENTER)

t1 = Label(voltage_frame, text="Units of V",bg="#ccef1b", font=("Arial", 10))
t1.pack()

textBox1=Text(voltage_frame, height=1, width=5)
textBox1.pack(side=tk.LEFT)

t2 = Label(voltage_frame, text=" to ",bg="#ccef1b", font=("Arial", 12))
t2.pack(side=tk.LEFT)

textBox2=Text(voltage_frame, height=1, width=5)
textBox2.pack(side=tk.LEFT)

###########################################################################################
temp_frame = LabelFrame(root, text = "e- Temperature",bg="#1BCCEF", font="Z003")
temp_frame.pack()
temp_frame.place(relx=.3,rely=.9,anchor=CENTER)

t3 = Label(temp_frame, text=" ",bg="#1BCCEF")
t3.pack(side=tk.LEFT)

textBox3=Text(temp_frame, height=1, width=5)
textBox3.pack(side=tk.LEFT)

t4 = Label(temp_frame, text=" ",bg="#1BCCEF")
t4.pack(side=tk.LEFT)

options = [ 
    "eV"
] 
  
clicked = StringVar() 
  
clicked.set( "eV" ) 
  
drop_temp = OptionMenu(temp_frame , clicked , *options ) 
drop_temp.pack(side=tk.LEFT) 

###########################################################################################

density_frame = LabelFrame(root, text = "e- Density",bg="#ccef1b", font="Z003")
density_frame.pack()
density_frame.place(relx=.5,rely=.9,anchor=CENTER)

t5 = Label(density_frame, text=" ",bg="#ccef1b")
t5.pack(side=tk.LEFT)

textBox4=Text(density_frame, height=1, width=5)
textBox4.pack(side=tk.LEFT)

t6 = Label(density_frame, text=" ",bg="#ccef1b")
t6.pack(side=tk.LEFT)

options = [ 
    "m^-3", 
] 
  
clicked = StringVar() 
  
clicked.set( "m^-3" ) 
  
drop_temp = OptionMenu(density_frame, clicked , *options ) 
drop_temp.pack(side=tk.LEFT) 

###########################################################################################

probe_length_frame = LabelFrame(root, text = "Probe Length",bg="#1BCCEF", font="Z003")
probe_length_frame.pack()
probe_length_frame.place(relx=.7,rely=.925,anchor=CENTER)

t7 = Label(probe_length_frame, text=" ",bg="#1BCCEF")
t7.pack(side=tk.LEFT)

textBox5=Text(probe_length_frame, height=1, width=5)
textBox5.pack(side=tk.LEFT)

t8 = Label(probe_length_frame, text=" ",bg="#1BCCEF")
t8.pack(side=tk.LEFT)

options = [ 
    "mm"
] 
  
# datatype of menu text 
clicked = StringVar() 
  
# initial menu text 
clicked.set( "mm" ) 
  
# create Dropdown menu 
drop_temp = OptionMenu(probe_length_frame, clicked , *options ) 
drop_temp.pack(side=tk.LEFT) 


###########################################################################################

probe_diameter_frame = LabelFrame(root, text = "Probe Diameter",bg="#1BCCEF", font="Z003")
probe_diameter_frame.pack()
probe_diameter_frame.place(relx=.7,rely=.85,anchor=CENTER)

t9 = Label(probe_diameter_frame, text=" ",bg="#1BCCEF")
t9.pack(side=tk.LEFT)

textBox6=Text(probe_diameter_frame, height=1, width=5)
textBox6.pack(side=tk.LEFT)

t10 = Label(probe_diameter_frame, text=" ",bg="#1BCCEF")
t10.pack(side=tk.LEFT)

options = [ 
    "mm"
] 
  
# datatype of menu text 
clicked = StringVar() 
  
# initial menu text 
clicked.set( "mm" ) 
  
# create Dropdown menu 
drop_temp = OptionMenu(probe_diameter_frame, clicked , *options ) 
drop_temp.pack(side=tk.LEFT) 

###########################################################################################

floating_potential_frame = LabelFrame(root, text = "Floating Potential",bg="#1BCCEF", font="Z003")
floating_potential_frame.pack()
floating_potential_frame.place(relx=.9,rely=.9,anchor=CENTER)

t13 = Label(floating_potential_frame, text=" ",bg="#1BCCEF")
t13.pack(side=tk.LEFT)

textBox7=Text(floating_potential_frame, height=1, width=5)
textBox7.pack(side=tk.LEFT)

t14 = Label(floating_potential_frame, text=" ",bg="#1BCCEF")
t14.pack(side=tk.LEFT)

options = [ 
    "V"
] 
  
# datatype of menu text 
clicked = StringVar() 
  
# initial menu text 
clicked.set( "V" ) 
  
# create Dropdown menu 
drop_temp = OptionMenu(floating_potential_frame, clicked , *options ) 
drop_temp.pack(side=tk.LEFT) 

##########################################################################################
#transition region voltage range inputs
transition_frame = LabelFrame(root, text = "Transition Range",bg="#ccef1b", font="Z003")
transition_frame.pack()
transition_frame.place(relx=.115,rely=.35,anchor=CENTER)

t14 = Label(transition_frame, text="Units of V",bg="#ccef1b", font=("Arial", 10))
t14.pack()

textbox8=Text(transition_frame, height=1, width=5)
textbox8.pack(side=tk.LEFT)

t15 = Label(transition_frame, text=" to ",bg="#ccef1b", font=("Arial", 12))
t15.pack(side=tk.LEFT)

textbox9=Text(transition_frame, height=1, width=5)
textbox9.pack(side=tk.LEFT)

###########################################################################################

t11 = Label(root, text="BRLPy",bg="#ef1bcc", font=("Arial Bold", 25))
t11.pack()
t11.place(relx=.5,rely=.05,anchor=CENTER)

t12 = Label(root, text="Russell Burns '25 Honors EPAD Project",bg="#ef1bcc", font=("Arial", 15))
t12.pack()
t12.place(relx=.5,rely=.1,anchor=CENTER)

def exitApp():
  root.destroy()

exitButton = Button(root, command = exitApp, text="Exit", bg ='red', highlightcolor="pink")
exitButton.pack(side=TOP)

###########################################################################################
fig = Figure(figsize=(5.5, 5), dpi=100)

plot_frame = None
canvas = None
toolbar = None

log_view_enabled = False  

def toggle_log_view():  
    global log_view_enabled
    log_view_enabled = not log_view_enabled
    plot()

def plot():
    global plot_frame, canvas, toolbar 
    
    if plot_frame is None:
        plot_frame = tk.Frame(root)
        plot_frame.pack()
        plot_frame.place(relx=.575,rely=.46,anchor=CENTER)

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack()

        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()
    fig.clear()

    try:
        v1 = float(textBox1.get(1.0, "end-1c"))
        v2 = float(textBox2.get(1.0, "end-1c"))
        T_ev = float(textBox3.get(1.0, "end-1c"))
        n_e = float(textBox4.get(1.0, "end-1c"))
        L = float(textBox5.get(1.0, "end-1c"))
        R = float(textBox6.get(1.0, "end-1c"))
        V_f = float(textBox7.get(1.0, "end-1c"))
    except ValueError:
        print("Error: Input values must be numbers.")
        return
    
    #R=0.8 #in units of mm
    #L=12.7 #in units mm
    #n_e = 1e14
    #T_ev = 1
    #V_f=0

    S = 2*np.pi*R*L*1e-6 # probe area im m^2 (NOT mm)
    Debye=(np.sqrt((8.854*10**-12*T_ev)/(n_e*e)))
    Xi=(R*.001)/Debye #R in m
    V_P = V_f - (T_ev * np.log(.6*np.sqrt(2*np.pi*(m_e/m_i))))

    #ion current calculations
    #Tiv = .1   # ion temperature
    #Va=np.sqrt((958000000000**1.62e-19*T_ev)/(2*np.pi*AN))
    #Jr=1.6E-19*S*Va
    #print(Jr)
    #I_is=Jr * (n_e/1000000) * 100000000000000 * 0.001    
    n_i=n_e
    I_is = .6 * e * n_i * np.sqrt(T_ev*e/m_i) * S


    A=1.12+(1/((1/(.00034*Xi**6.87))-(1/(.145*np.log(Xi/110)))))
    B=.5+.008*Xi**1.5/np.exp(.18*Xi**.80)
    C=1.07+.95/Xi**1.01
    D=.05+1.54*Xi**.3/np.exp(1.135*Xi**.370)

    T_iv=.04
    def Ii(VB):
        if VB < V_f:
            return -(1 / ((1 / (A * (((V_f-VB)/T_iv))**B)**4 + 1 / (C * (((V_f-VB)/T_iv))**D)**4)**(1/4))) * I_is
        else:
            return 0
            #return -I_is * np.exp((V_P - VB) / T_iv)
        




    #electron current calculations

    def gamma(Xi, a, lambd, r):
        return (-a * (lambd**r) / special.gamma(r)) * (Xi**(r-1)) * np.exp(-lambd * Xi)

    def electron_param(x, Xi):
        base = (2 / np.sqrt(np.pi)) * np.sqrt(x) + special.erfcx(np.sqrt(x))
        return (((base-1) * np.exp(Xi * gamma(Xi, 4.17, .05307, 1.168) * x**gamma(Xi, -0.8655, .1507, 2.3)))+1)


    def coeft(x):
        result = (np.sqrt(np.pi) / 2) * special.erfcx(x) ##special function for exp(x**2) * erfcx(x), which is simply g(x) as defined in the paper
        return result


    def eta(y):
        return (1 / np.sqrt(np.pi)) * coeft(np.sqrt(y))

    def inner_integral(y_prime):
        result, _ = quad(eta, 0, y_prime)
        return result

    def outer_integral(y):
        def integrand(y_prime):
            inner = inner_integral(y_prime)
            if inner == 0:
                return 0
            return 1 / np.sqrt(2 * inner)

        result, _ = quad(integrand, 0, y)
        return result

    outer_integral_vectorized = np.vectorize(outer_integral)

    def planar_electron(VB,Xi):
        return (1 + (outer_integral_vectorized(VB)/Xi))

    I_es = e * n_e * S * np.sqrt((T_ev*e)/(2*np.pi*m_e)) #chen

    def Ie(VB):
       if VB < V_P:
            return I_es * np.exp(-(V_P - VB) / T_ev)
       else:
        if Xi >= 21.316:
            return I_es * planar_electron((V_P - VB),Xi)
        elif Xi < 21.316:
           return I_es * electron_param((VB - V_P),Xi) 



    VB_range = np.linspace(v1, v2, 100)
    global ideal_current 
    ideal_current = np.array([Ie(VB) for VB in VB_range]) + np.array([Ii(VB) for VB in VB_range])
    plot1 = fig.add_subplot(111)
    fig.suptitle("IV Sweep with BRL Fit")
    fig.supxlabel("Probe Voltage (V)")
    fig.supylabel("Current (A)")
    plot1.tick_params(axis='y', labelsize=8) 
    #plot1.plot(VB_range, ideal_current, color='blue', linestyle='-', linewidth=2, label = "Ideal Sweep")
    #plot1.plot(VB_range, np.array([Ie(VB) for VB in VB_range]), color='red', linestyle='-', linewidth=2, label = "Electron")
    #plot1.plot(VB_range, np.array([Ii(VB) for VB in VB_range]), color='green', linestyle='-', linewidth=2, label = "Ion")

    if log_view_enabled:
        plot1.plot(VB_range, np.abs(ideal_current), color='blue', linestyle='-', linewidth=2, label="BRL Curve")
    else:
        plot1.plot(VB_range, ideal_current, color='blue', linestyle='-', linewidth=2, label="BRL Curve")

    plot1.set_yscale('log' if log_view_enabled else 'linear')

    if imported_data is not None:
        y_data = np.abs(imported_data[:,1]) if log_view_enabled else imported_data[:,1] 
        plot1.scatter(imported_data[:,0], y_data, color="red", label="Imported Data")
    
    plot1.grid(True)
    fig.legend(fontsize=8, loc="upper right")
    canvas.draw()

def extract_parameters():
    try:
        vmin = float(textbox8.get(1.0, "end-1c"))
        vmax = float(textbox9.get(1.0, "end-1c"))
        Te_guess = float(textBox3.get(1.0, "end-1c"))
        L = float(textBox5.get(1.0, "end-1c"))
        R = float(textBox6.get(1.0, "end-1c"))
    except ValueError:
        print("Please enter valid numbers.")
        return

    if imported_data is None:
        print("No data imported.")
        return

    voltages = imported_data[:, 0]
    currents = imported_data[:, 1]

    #finds nearest values
    mask = (voltages >= vmin) & (voltages <= vmax)
    V_fit = voltages[mask]
    I_fit = currents[mask]

    # Use absolute value of current and log-transform
    with np.errstate(divide='ignore'):
        ln_I_fit = np.log(np.abs(I_fit))

    # Remove -inf from log(0) if any
    mask_finite = np.isfinite(ln_I_fit)
    V_fit = V_fit[mask_finite]
    ln_I_fit = ln_I_fit[mask_finite]

    # Linear fit to ln(I) vs V
    slope, intercept, r_value, p_value, std_err = linregress(V_fit, ln_I_fit)

    #te estimate
    Te_fit = ((1 / slope) + Te_guess)/2
    Te_uncertainty = abs(Te_fit - Te_guess)
    print(f"Te: {Te_fit:.4f} pm {Te_uncertainty:.4f} eV")

    #vf estimate
    V_f_guess=float(textBox7.get(1.0, "end-1c"))
    V_P_fit = (V_f_guess - (Te_fit * np.log(.6*np.sqrt(2*np.pi*(m_e/m_i)))))
    V_P_uncertainty = abs(((Te_fit+Te_uncertainty) * np.log(.6*np.sqrt(2*np.pi*(m_e/m_i))))-((Te_fit) * np.log(.6*np.sqrt(2*np.pi*(m_e/m_i)))))
    print(f"V_P: {V_P_fit:.4f} pm {V_P_uncertainty:.4f} V")


    # Get the corresponding current values
    I_es_fit = currents[(np.abs(voltages - V_P_fit)).argmin()]
    I_es_high = currents[(np.abs(voltages - (V_P_fit+V_P_uncertainty))).argmin()] 
    I_es_low = currents[(np.abs(voltages - (V_P_fit-V_P_uncertainty))).argmin()]

    S = 2*np.pi*R*L*1e-6 # probe area im m^2 (NOT mm)

    def calc_ne(I_es, Te):
        return I_es / (e * S * np.sqrt((Te * e) / (2 * np.pi * m_e)))

    n_e_fit = calc_ne(I_es_fit, Te_fit)
    n_e_high = calc_ne(I_es_high, Te_fit + Te_uncertainty)
    n_e_low = calc_ne(I_es_high, Te_fit - Te_uncertainty)
    n_e_uncertainty_high = abs(n_e_high - n_e_fit)
    n_e_uncertainty_low = abs(n_e_low - n_e_fit)
    n_e_fit_2=(n_e_fit+n_e_uncertainty_high-n_e_uncertainty_low)
    n_e_uncertainty=((n_e_fit_2+n_e_uncertainty_high)-(n_e_fit_2-n_e_uncertainty_low))/2

    print(f"n_e: {n_e_fit_2:.3e} pm {n_e_uncertainty:.3e} m^-3")

    #Debye=(np.sqrt((8.854*10**-12*T_ev)/(n_e*e)))

    Debye_fit=(np.sqrt((8.854*10**-12*Te_fit)/(n_e_fit_2*e)))
    Debye_uncertainty=abs(Debye_fit-(np.sqrt((8.854*10**-12*(Te_fit+Te_uncertainty))/((n_e_fit_2-n_e_uncertainty)*e))))
    print(f"Debye: {Debye_fit:.3e} pm {Debye_uncertainty:.3e} m")
    Xi_fit=(R*.001)/Debye_fit #R in m
    Xi_uncertainty=abs(Xi_fit-((R*.001)/(Debye_fit+Debye_uncertainty)))
    print(f"Xi: {Xi_fit:.3e} pm {Xi_uncertainty:.3e}")

    

log_button = Button(root, text="Log View", height=2, width=10, command=toggle_log_view)
log_button.pack()
log_button.place(relx=.115, rely=.5, anchor=CENTER)

plot_button = Button(root, command = plot, height = 2, width = 10, text="Plot")
plot_button.pack()
plot_button.place(relx=.115,rely=.75,anchor=CENTER)

extract_button = Button(root, text="Extract", height = 2, width = 10, command=extract_parameters)
extract_button.pack()
extract_button.place(relx=.115, rely=.25, anchor=CENTER)


### curve fit assessment
def import_csv():
    global imported_data
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")])
    if file_path.endswith(".csv"):
        try:
            df = pd.read_csv(file_path)
            imported_data = df.to_numpy()
            plot()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")
    if file_path.endswith(".txt"):
        try:
            imported_data = np.loadtxt(file_path, skiprows=22)
            plot()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load TXT: {str(e)}")
    if not file_path:
        return


#def calculate_rmse():
    #interp_generated = np.interp(imported_data[:,0], ideal_current[:,0], ideal_current[:,1])
   # rmse = np.sqrt(np.mean((imported_data[:,1] - interp_generated) ** 2))
    #rmse_label.config(text=f"RMSE: {rmse:.6f}")


btn_import = tk.Button(root, text="Import Data", height = 2, width = 10, command=import_csv)
btn_import.pack()
btn_import.place(relx=.115,rely=.65,anchor=CENTER)

#btn_rmse = tk.Button(root, text="Calculate RMSE", command=calculate_rmse)
#btn_rmse.pack()
#btn_rmse.place(relx=.1,rely=.55,anchor=CENTER)

#rmse_label = tk.Label(root, text="RMSE: ")
#rmse_label.pack()
#rmse_label.place(relx=.1,rely=.65,anchor=CENTER)

root.mainloop()																																			
