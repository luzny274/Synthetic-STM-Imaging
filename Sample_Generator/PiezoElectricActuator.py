import torch
import matplotlib.pyplot as plt

# Define the Bouc-Wen differential equation
def bouc_wen_diff(t, z, h, voltage_f, actuator):
    dv_dt = (voltage_f(t+h/2, actuator) - voltage_f(t-h/2, actuator)) / h #We approximate that the voltage changes in a linear way
    dz_dt = actuator.A * dv_dt - actuator.beta * abs(dv_dt) * z - actuator.gamma * dv_dt * abs(z)
    
    return dz_dt

def rungeKutta(xs, y0, dy_dx, voltage_f, actuator):
    y0 = y0 if torch.is_tensor(y0) else torch.tensor(y0)
    ys_shape = (xs.shape[0]) if y0.ndim < 1 else (xs.shape[0],) +  y0.shape
    
    ys = torch.zeros(ys_shape)
    ys[0] = y0
    
    for i in range(len(xs) - 1):
        h = xs[i+1] - xs[i]
        k1 = h * dy_dx(xs[i]        , ys[i]         , 0.5 * h, voltage_f, actuator)
        k2 = h * dy_dx(xs[i] + 0.5*h, ys[i] + 0.5*k1, 0.5 * h, voltage_f, actuator)
        k3 = h * dy_dx(xs[i] + 0.5*h, ys[i] + 0.5*k2, 0.5 * h, voltage_f, actuator)
        k4 = h * dy_dx(xs[i] + h    , ys[i] + k3    , 0.5 * h, voltage_f, actuator)
  
        ys[i+1] = ys[i] + (1.0/6.0)*(k1 + 2*k2 + 2*k3 + k4)
  
    return ys

class PiezoElectricActuator:
    def calibrate(self, plot_bool):
        # Time range for simulation
        t_span = (0, 4)
        t_eval = torch.linspace(t_span[0], t_span[1], 50)

        def voltage_f(t, actuator): # Triangle wave voltage
            m = torch.remainder(t + 1.5, 1)  # Get fractional part (equivalent to np.modf()[0])
            return torch.where(((t + 1.5).to(torch.int) % 2).bool(), 1 - m, m) - 0.5
    
        z0 = 0

        v = voltage_f(t_eval, self)
        z = rungeKutta(t_eval, z0, bouc_wen_diff, voltage_f, self)


        x = v - z

        self.linear_slope = (torch.max(x) - torch.min(x)) / (torch.max(v) - torch.min(v))
        self.x_range = (torch.max(x) - torch.min(x))

        fig = None

        if plot_bool:
            fig, ax = plt.subplots(figsize=(10, 5))  
            sc = ax.scatter(v.cpu(), x.cpu() / self.x_range.cpu(), c=t_eval.cpu(), cmap='plasma', marker='o')
            cbar = fig.colorbar(sc)
            cbar.set_label('Time')
            
            ax.set_xlabel('Voltage')
            ax.set_ylabel('Displacement')
            ax.set_title('Bouc-Wen Model for Piezoelectric Actuator')
            
        return fig
        
    def __init__(self, A, beta, gamma, x0, plot_bool):
        self.A     = A
        self.beta  = beta
        self.gamma = gamma

        self.calibrate(plot_bool)

        x0 = x0 if torch.is_tensor(x0) else torch.tensor(x0)
        self.v = x0 * self.x_range
        self.z = torch.zeros_like(self.v)

        self.vs = self.v[None].clone().detach()
        self.xs = x0[None]
        self.ts = torch.tensor([0.0]).view((1))

    def move(self, desired_change, duration):
        desired_change = desired_change if torch.is_tensor(desired_change) else torch.tensor(desired_change)
        self.desired_change = desired_change[None]
        
        self.duration = duration

        steps = 10

        t_span = (0, duration)
        t_eval = torch.linspace(t_span[0], t_span[1], steps)

        def voltage_f(t, actuator): #Ramp voltage
            t = t if torch.is_tensor(t) else torch.tensor(t)
            t = t if t.ndim >= 1        else t[None]
            t_shape = (t.shape[0],) + (1,) * (self.desired_change.ndim - 1)
            t = t.view(t_shape)
            
            return torch.where((t < 0).bool(), actuator.v, actuator.v + actuator.desired_change / actuator.duration * t)
            
        v = voltage_f(t_eval, self)
        z = rungeKutta(t_eval, self.z, bouc_wen_diff, voltage_f, self)

        x = v - z
        self.v = v[-1]
        self.z = z[-1]

        self.vs = torch.concatenate((self.vs, v), axis=0)
        self.xs = torch.concatenate((self.xs, x / self.x_range), axis=0)

        self.ts = torch.concatenate((self.ts, self.ts[-1] + t_eval + duration / steps), axis=0)

        return self.xs[-1]