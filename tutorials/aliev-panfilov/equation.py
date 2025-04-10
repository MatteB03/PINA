import numpy as np
import matplotlib.pyplot as plt

def laplacian(V, h):
    """
    Calcola il laplaciano (derivata seconda) per un array 1D V usando differenze finite centrali.
    """
    d2V = np.zeros_like(V)
    d2V[1:-1] = (V[:-2] - 2 * V[1:-1] + V[2:]) / (h ** 2)
    return d2V

def AlPan(y, Istim):
    """
    Calcola le derivate temporali per il sistema di equazioni di Aliev-Panfilov.
    
    Parametri:
      y: array di shape (X,2), dove la prima colonna è V e la seconda è W.
      Istim: array di stimolo esterno (dimensione X).
      
    Restituisce:
      dydt: array di shape (X,2) contenente [dV/dt, dW/dt].
    """
    # Parametri del modello
    a = 0.01
    k = 8.0
    mu1 = 0.2
    mu2 = 0.3
    epsi = 0.002
    b = 0.15
    h_cell = 0.1  # lunghezza della cella (unità spaziali)
    D = 0.1       # coefficiente di diffusione

    V = y[:, 0]
    W = y[:, 1]
    
    # Calcola il laplaciano per V (diffusione)
    dV_diff = 4 * D * laplacian(V, h_cell)
    
    # Equazione per W
    dWdt = (epsi + mu1 * W / (mu2 + V)) * (-W - k * V * (V - b - 1))
    
    # Equazione per V (nonlineare, diffusione e stimolo)
    dVdt = -k * V * (V - a) * (V - 1) - W * V + dV_diff + Istim
    
    dydt = np.column_stack((dVdt, dWdt))
    return dydt


def AlievPanfilov1D_RK_Istim():
    """
    Simula il modello di Aliev-Panfilov in 1D usando il metodo Runge-Kutta 4 (RK4).
    
    Parametri:
      BCL      : Basic Cycle Length (AU), intervallo temporale tra stimoli.
      ncyc     : Numero di cicli (stimolazioni).
      extra    : Tempo aggiuntivo dopo BCL * ncyc (AU).
      ncells   : Numero di celle interne (lunghezza del cavo 1D).
      iscyclic : False per cavo (condizioni al bordo di Neumann), True per anello (condizioni periodiche, non implementate completamente).
      flagmovie: True per mostrare una visualizzazione in tempo reale.
      
    Restituisce:
      Vsav, Wsav: Array con le soluzioni di V e W salvate per il plotting.
    """
    # Parametri di esempio
    BCL = 70        # Basic Cycle Length
    ncyc = 1         # Numero di cicli (stimoli)
    extra = 0        # Tempo extra dopo ncyc * BCL
    ncells = 70     # Numero di celle interne
    iscyclic = False # Cavo (False) o anello (True)
    flagmovie = True # Visualizzazione in tempo reale
    # Estensione del dominio per includere le celle di bordo
    X = ncells + 2  # includiamo due celle extra per i BC
    
    # Definizione della geometria di stimolazione: stimola le prime 5 celle interne
    stimgeo = np.zeros(X, dtype=bool)
    stimgeo[:5] = True
    
    # Parametri temporali
    dt = 0.005  # passo temporale
    gathert = int(round(1 / dt))  # salva ogni 1 AU
    tend = BCL * ncyc + extra  # durata totale della simulazione (AU)
    stimdur = 1  # durata dello stimolo (AU)
    
    # Definisci lo stimolo: quando stimolato, Ia = 0.1; altrimenti 0
    Ia = 0.1 * stimgeo.astype(float)
    
    # Condizioni iniziali: V e W inizialmente uguali a 0.01 in tutto il dominio
    V = np.full(X, 0.01)
    W = np.full(X, 0.01)
    
    # Prealloca array per salvare le soluzioni (solo celle interne)
    n_save = int(np.ceil(tend / dt / gathert))
    Vsav = np.zeros((ncells, n_save))
    Wsav = np.zeros((ncells, n_save))
    
    ind = 0   # contatore delle iterazioni
    kk = 0    # contatore per il numero di stimoli applicati
    Istim = np.zeros(X)
    
    # Inizializza lo stato y (ogni riga contiene [V, W])
    y = np.column_stack((V, W))
    
    # Setup per il movie se flagmovie è True
    #if flagmovie:
    #    plt.ion()
    #    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    
    t_values = np.arange(dt, tend + dt, dt)
    sav_index = 0
    
    # Ciclo temporale con integrazione RK4
    for t in t_values:
        ind += 1
        # Applica lo stimolo se siamo nel tempo appropriato
        if t >= BCL * kk and kk < ncyc:
            Istim = Ia.copy()
        # Dopo stimdur, interrompi lo stimolo e incrementa il contatore
        if t >= BCL * kk + stimdur * 2:
            kk += 1
            Istim = np.zeros(X)
        
        # Calcola i coefficienti RK4
        k1 = AlPan(y, Istim)
        k2 = AlPan(y + dt / 2 * k1, Istim)
        k3 = AlPan(y + dt / 2 * k2, Istim)
        k4 = AlPan(y + dt * k3, Istim)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # Estrai V e W dall'array aggiornato
        V = y[:, 0]
        W = y[:, 1]
        
        # Applica le condizioni al bordo: condizioni di Neumann per V (cavo 1D)
        if not iscyclic:
            V[0] = V[1]
            V[-1] = V[-2]
        else:
            # Per un anello, qui andrebbero implementate condizioni periodiche
            pass
        
        # Salva i dati ogni gathert iterazioni
        if ind % gathert == 0:
            Vsav[:, sav_index] = V[1:-1]  # esclude le celle di bordo
            Wsav[:, sav_index] = W[1:-1]
            sav_index += 1
            
            # Se flagmovie è True, mostra un movie della propagazione
            #if flagmovie:
            #    ax1.clear()
            #    V_display = np.tile(V[1:-1], (max(ncells // 20, 1), 1))
            #    im1 = ax1.imshow(V_display, vmin=0, vmax=1, aspect='auto', cmap='jet')
            #    ax1.set_title(f'V (AU) - Time: {t:.0f} AU')
            #    ax1.set_xlabel('x (cells)')
            #    fig.colorbar(im1, ax=ax1)
                
            #    ax2.clear()
            #    W_display = np.tile(W[1:-1], (max(ncells // 20, 1), 1))
            #    im2 = ax2.imshow(W_display, vmin=0, vmax=1, aspect='auto', cmap='jet')
            #    ax2.set_title(f'W (AU) - Time: {t:.0f} AU')
            #    ax2.set_xlabel('x (cells)')
            #    fig.colorbar(im2, ax=ax2)
                
            #    plt.pause(0.01)
    
    #if flagmovie:
    #    plt.ioff()
    #    plt.show()
    return Vsav, Wsav
    # Plot finale delle soluzioni salvate
"""     plt.figure(figsize=(10, 8))
    
    

    plt.subplot(2, 1, 1)
    plt.title('V (AU)')
    plt.imshow(Vsav.T, aspect='auto', origin='lower', vmin=0, vmax=1, cmap='jet')
    plt.ylabel('Time (AU)')
    plt.xlabel('x (cells)')
    plt.colorbar()
    
    plt.subplot(2, 1, 2)
    plt.title('W (AU)')
    plt.imshow(Wsav.T, aspect='auto', origin='lower', vmin=0, vmax=2.4, cmap='jet')
    plt.ylabel('Time (AU)')
    plt.xlabel('x (cells)')
    plt.colorbar()
    
    #plt.show()
    plt.savefig("./plot_finale.png") """