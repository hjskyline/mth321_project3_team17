"""Figures from recorded measurements; GIF panels use the same fixed axes."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .heat import evolve, grid, rate
from .studies import ALL, BASE, STARTUP_TOL, profile

COLORS = {"FTCS": "#246490", "CN": "#a94b26", "Rannacher": "#147466"}


def finish(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.facecolor": "white"})


def panels(title, subtitles, figsize=(11.5, 5.0)):
    fig, axes = plt.subplots(1, len(subtitles), figsize=figsize, layout="constrained")
    fig.suptitle(title, fontsize=19, weight="bold", color="#182c3a")
    axes = np.atleast_1d(axes)
    for ax, label in zip(axes, subtitles):
        ax.set_title(label, loc="left", pad=12)
    return fig, axes


def convergence_figure(rows, path):
    fig, axes = panels("Halving the step reveals the order of accuracy",
                       ["Spatial error · continuous reference", "Time error · fixed-grid reference"])
    for ax, study, variable in zip(axes, ("space","time"), ("h","dt")):
        methods = BASE if study == "space" else ALL
        for name, marker, style in zip(methods, ("o","x","s"), ("-","--","-.")):
            g = [r for r in rows if r["study"]==study and r["method"]==name]
            p = g[-1]["pairwise_order"]
            ax.loglog([r[variable] for r in g], [r["l2_error"] for r in g],
                      linestyle=style, marker=marker, ms=8, mfc="white", color=COLORS[name],
                      label=f"{name}: {2**p:.1f}× smaller error per halving")
        ax.set(xlabel="Grid spacing h" if study=="space" else "Time step Δt", ylabel="Discrete L2 error")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=10)
    finish(fig,path)


def stability_figure(rows, path):
    fig, axes = panels("At z = 2, FTCS reaches its stability limit and CN changes sign",
                       ["FTCS · after 12 steps", "CN · after the first step"])
    for ax, method, step in zip(axes,BASE,(12,1)):
        g = [r for r in rows if r["mode"]==63 and r["method"]==method and r["step"]==step]
        x = [r["z_high"] for r in g]
        y = [r["signed_amplitude"] for r in g]
        ax.axvline(2,ls=":",color="#666666")
        ax.axhline(1 if method=="FTCS" else 0,ls="--",color="#666666",
                   label="Initial amplitude" if method=="FTCS" else "Sign change")
        ax.plot(x,y,"o-",lw=2,ms=7,color=COLORS[method])
        ax.set(xlabel="Highest-mode step size z = μ Δt", ylabel="Mode amplitude",
               xlim=(1.75,2.25))
        ax.annotate(f"{y[-1]:+.3f}",(x[-1],y[-1]),xytext=(-4,12),
                    textcoords="offset points",ha="right",weight="bold")
        ax.margins(y=0.2)
        ax.legend(fontsize=10)
    finish(fig,path)


def startup_figure(rows, path):
    from matplotlib.colors import ListedColormap

    fig,axes = panels("When does the first heat step need damping?",
                      ["Decision at each setting", "Outcomes across both high modes"], (12,4.6))
    zs = sorted({r["z_high"] for r in rows})
    weights = sorted({r["weight"] for r in rows},reverse=True)
    colors = ["#d4e9f2","#fcce74","#974276"]
    labels = ["CN","Damp","Refine"]
    maps = []
    for k in (15,31):
        matrix = []
        for a in weights:
            line = []
            for z in zs:
                passes = {r["method"]:r["meets_threshold"] for r in rows
                          if r["weight"]==a and r["z_high"]==z and r["high_mode"]==k}
                line.append(0 if passes["CN"] else 1 if passes["Rannacher"] else 2)
            matrix.append(line)
        maps.append(matrix)
    ax,counts=axes
    ax.imshow(maps[0],cmap=ListedColormap(colors),vmin=-0.5,vmax=2.5,aspect="auto")
    for i,line in enumerate(maps[0]):
        for j,case in enumerate(line):
            ax.text(j,i,labels[case],ha="center",va="center",fontsize=10,
                    color="white" if case==2 else "#182c3a")
    ax.set_xticks(range(len(zs)),[f"{z:g}" for z in zs])
    ax.set_yticks(range(len(weights)),[f"{w:g}" for w in weights])
    same = np.array_equal(*maps)
    ax.set(xlabel="High-mode step size z",ylabel="Initial high-mode weight")
    ax.set_title("Modes 15 and 31 give the same map" if same else "Decision map for mode 15",
                 loc="left",pad=12)
    totals=np.bincount(np.ravel(maps),minlength=3)
    counts.barh(range(3),totals,color=colors)
    counts.set_yticks(range(3),["Ordinary CN","Use damping","Refine step"])
    counts.invert_yaxis()
    counts.set(xlabel="Tested cases",xlim=(0,max(totals)*1.24))
    for i,num in enumerate(totals):
        counts.text(num+0.4,i,f"{num} / {sum(totals)}",va="center",weight="bold")
    fig.supxlabel("First-step L2 tolerance 0.01   ·   Damp = two backward Euler half-steps   ·   Refine = neither full-step option passes",
                  fontsize=11)
    finish(fig,path)


def cost_figure(rows, matched, path):
    targets = sorted({r["target"] for r in matched},reverse=True)
    fig,axes=panels("How long does each method take at the same accuracy?",
                    [f"Maximum error ≤ {t:g}" for t in targets])
    for ax,t in zip(axes,targets):
        for i,name in enumerate(ALL):
            win = next(r for r in matched if r["target"]==t and r["method"]==name)
            if not win["eligible"]:
                ax.text(0,i,"Target missed",va="center")
                continue
            trial = next(r for r in rows if r["method"]==name and r["steps"]==win["steps"])
            v = win["median_ms"]
            ax.barh(i,v,color=COLORS[name],height=0.52)
            ax.errorbar(v,i,xerr=[[v-trial["min_ms"]],[trial["max_ms"]-v]],
                        fmt="none",ecolor="#25303a",capsize=3)
            ax.annotate(f"{v:.2f} ms · {win['steps']} steps",(trial["max_ms"],i),
                        xytext=(5,0),textcoords="offset points",va="center",fontsize=10)
        ax.set_yticks(range(3),ALL)
        ax.invert_yaxis()
        ax.set_xlim(0,ax.get_xlim()[1]*1.48)
        ax.set_xlabel("Solver time (ms)")
    fig.supxlabel("Fastest tested configuration meeting each target · five timed runs · whiskers show the range",fontsize=11)
    finish(fig,path)


def pulse_figure(points, metrics, path):
    fig,(left,right)=panels("Startup damping smooths the edge of a square heat pulse",
                            ["Temperature near the left jump · step 1","Maximum error against the continuous PDE"])
    rows=[r for r in points if r["step"]==1 and 0.27<=r["x"]<=0.48]
    for key,label,color,style in (("initial","Initial","0.65",":"),("exact","Exact PDE","#222222","--"),
                                  ("cn","CN",COLORS["CN"],"o-"),("damped","Rannacher",COLORS["Rannacher"],".-")):
        left.plot([r["x"] for r in rows],[r[key] for r in rows],style,color=color,label=label,ms=4)
    left.set(xlabel="Position along rod",ylabel="Temperature")
    left.legend(fontsize=10)
    for method in ("CN","Rannacher"):
        g=[r for r in metrics if r["method"]==method]
        right.semilogy([r["step"] for r in g],[r["linf_pde"] for r in g],"o-",
                       color=COLORS[method],label=method)
    right.set(xticks=[1,2,4,8],xlabel="Full time steps",ylabel="Maximum nodal error")
    right.legend()
    first={r["method"]:r["linf_pde"] for r in metrics if r["step"]==1}
    fig.supxlabel(f"Step 1 maximum error: CN {first['CN']:.3f}   ·   Rannacher {first['Rannacher']:.3f}   ·   "
                  f"{first['CN']/first['Rannacher']:.1f}× reduction   |   128 intervals, Δt = 1/1024",fontsize=12)
    finish(fig,path)


def positivity_figure(rows,path):
    fig,axes=panels("A stable update can produce a negative temperature",
                    ["Temperature at the original spike","Fraction of initial squared L2 norm"])
    g=[r for r in rows if r["method"]=="CN"]
    axes[0].plot([r["r"] for r in g],[r["reference_centre"] for r in g],"k--",label="Exact grid evolution")
    for ax,key in zip(axes,("centre","energy_ratio")):
        for name in ("CN","Rannacher"):
            g=[r for r in rows if r["method"]==name]
            ax.plot([r["r"] for r in g],[r[key] for r in g],"o-",color=COLORS[name],label=name)
        ax.set_xscale("log",base=2)
        ax.set_xticks([0.25,0.5,1,2,4,8,16],["¼","½","1","2","4","8","16"])
        ax.set(xlabel="Mesh ratio r = Δt / h²",ylabel="Centre temperature" if key=="centre" else "Energy ratio")
        ax.axhline(0 if key=="centre" else 1,color="0.55",ls=":")
        ax.legend(fontsize=10)
    v=next(r["centre"] for r in rows if r["method"]=="CN" and r["r"]==16)
    axes[0].annotate(f"CN {v:+.3f}",(16,v),xytext=(-35,4),textcoords="offset points",
                     ha="right",color=COLORS["CN"],weight="bold",
                     bbox=dict(facecolor="white",edgecolor="none",pad=1))
    fig.supxlabel("Unit-height spike on a zero background · first full interval · exact heat flow preserves non-negativity",fontsize=11)
    finish(fig,path)


def transfer_figure(rows, cost, path):
    from .transfer import PROFILES
    fig,(a,b)=panels("A tolerance-driven startup transfers to three new heat profiles",
                     ["Rule chosen across 81 cases", "Solve counts at matched first-step accuracy"])
    categories=("Ordinary CN","Damped","Split CN")
    colors=(COLORS["CN"],COLORS["Rannacher"],"#47659a")
    for i,name in enumerate(PROFILES):
        group=[r for r in rows if r["profile"]==name]
        counts=[sum(r["selected"]=="CN" and r["first_parts"]==1 for r in group),
                sum(r["selected"]=="Rannacher" for r in group),
                sum(r["first_parts"]>1 for r in group)]
        start=0
        for count,color,label in zip(counts,colors,categories):
            a.barh(i,count,left=start,color=color,label=label if i==0 else None)
            if count:
                a.text(start+count/2,i,str(count),va="center",ha="center",color="white",weight="bold")
            start+=count
    a.set_yticks(range(3),PROFILES)
    a.set(xlabel="Cases (27 per profile)",xlim=(0,27))
    a.legend(loc="lower center",bbox_to_anchor=(0.5,-0.35),ncol=3,fontsize=9)
    for i,tol in enumerate(sorted({r["tolerance"] for r in cost})):
        for shift,rule,color in [(-0.19,"Uniform CN","#77828e"),(0.19,"Controlled","#47659a")]:
            r=next(r for r in cost if r["tolerance"]==tol and r["policy"]==rule)
            b.barh(i+shift,r["total_solves"],height=0.32,color=color,label=rule if i==0 else None)
            b.text(r["total_solves"]+4,i+shift,str(r["total_solves"]),va="center",fontsize=10)
    b.set_yticks(range(3),[f"{t:g}" for t in sorted({r["tolerance"] for r in cost})])
    b.set(xlabel="Linear solves across 27 inputs",ylabel="L2 tolerance")
    b.set_xlim(0,b.get_xlim()[1]*1.08)
    b.legend(loc="lower center",bbox_to_anchor=(0.5,-0.35),ncol=2,fontsize=10)
    finish(fig,path)


def transient(path):
    m, k, weight, steps = 64, 31, 0.25, 8
    dt = 12.0 / rate(m, k)
    components = [(1, 1), (k, weight)]
    states = {n: evolve(profile(m, components), dt, steps, n, range(steps+1))
              for n in ("CN", "Rannacher")}
    # Remove the exact semidiscrete low mode so the high-frequency transient is visible.
    x = np.r_[0, grid(m), 1]
    fig, ax = plt.subplots(figsize=(6.6, 3.8), layout="constrained")
    lines = {n: ax.plot([], [], color=c, label=n)[0] for n,c in
             (("Exact grid evolution", "0.15"), ("CN", COLORS["CN"]), ("Rannacher", COLORS["Rannacher"]))}
    lines["Exact grid evolution"].set_linestyle("--")
    ax.set(xlim=(0,1), ylim=(-0.28,0.28), xlabel="Position x", ylabel="Temperature minus low mode")
    ax.set_yscale("asinh", linear_width=1e-4)
    ax.legend(loc="lower right", fontsize=8)

    def frame(i):
        t = i*dt
        low = profile(m, [(1,1)], t, discrete=True)
        exact = profile(m, components, t, discrete=True)
        for name,line in lines.items():
            u = exact if name == "Exact grid evolution" else states[name][i]
            line.set_data(x, np.r_[0, u-low, 0])
        ax.set_title(f"High-frequency transient  |  step {i}  |  t = {t:.5f}")
        return list(lines.values())

    animation = FuncAnimation(fig, frame, frames=steps+1, interval=500, repeat=True)
    animation.save(path, writer=PillowWriter(fps=2), dpi=90)
    plt.close(fig)
