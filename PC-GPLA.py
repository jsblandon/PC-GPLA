"""Plasma Computacional Grupo Plasma, Laser y Aplicaciones"""

"""Simulacion de plasma electrostatico (1D) y de plasma bajo la accion de un campo magnetico uniforme (1.5D2V)"""

"""Autores: Juan Sebastian Blandon Luengas, Juan Pablo Grisales Campeon"""

# Se importan las librerias necesarias:
import scipy as sp         # Operaciones matematicas y manejo de Arreglos
import numpy as np         # 
import pylab as plt        # Graficas 
import sys                 # Libreria de la funcion de salida
import os                  # Libreria para manipular archivos
import shutil              # Libreria para copiar, mover y renombrar archivos
from mpl_toolkits.mplot3d import Axes3D# Modulo para graficar en tres dimensiones
import matplotlib.ticker as mtick           # Modulo para la modificacion de las
                                            # etiquetas de los ejes
from scipy.fftpack import fft2              # FFT dos dimensional
from scipy.fftpack import fftshift as shift # Corrimiento al cero 

#===================================================================
# PRIMERA ETAPA: Generacion de la malla (cargar las posiciones de las particulas)
#===================================================================
def cargarposicion():            
    global longitud_malla, rho0, nparticulas, carga_masa, aP
    global carga,masa,pared_izquierda,pared_derecha
    print "Se cargan las particulas"

    # Se define la Malla Euleriana
    plasma_li = 0.
    plasma_lf = longitud_malla
    pared_izquierda = 0.
    pared_derecha = longitud_malla
      
    # Longitud para cargar las particulas  
    dimension_plasma = plasma_lf - plasma_li  
    espacio_particulas = dimension_plasma/nparticulas                    
    # Carga normalizada de la pseudo-particula para obtener ncrit=1 (rhoc=-1) 
    carga = -rho0*espacio_particulas         
    # Relacion carga-masa (necesaria para el diagnostico de la energia K) 
    masa = carga/carga_masa  

    for i in range(nparticulas):
        """ Ciclo que ubica las particulas en el rango deseado"""
        x[i] = plasma_li + espacio_particulas*(i+0.5) # Metodo Leap-Frog sobre la posicion
        x[i] += aP*sp.cos(x[i]) # Perturbacion

    return True
        
#==================================================================
#  SEGUNDA ETAPA: carga de las velocidades de las particulas
#==================================================================    
def cargarvelocidad(tdistribucion,vh):       
    global nparticulas,v,longitud_malla,vP,F,Bo,vx,vy
    print "Configuracion inicial de la distribucion de velocidad"

    if(tdistribucion == 0):  
        """Plasma frio"""
        v[1:nparticulas] = 0.
        F = sp.zeros(nparticulas)

    elif ( tdistribucion == 1):
        """Inestabilidad Two-Stream : Se asume una distribucion de Maxwell para dos flujos de electrones"""
        F = sp.zeros(nparticulas)
        
        for i in range(nparticulas):    
            fmax = 0.5 * (1. + sp.exp(-2. * vh * vh))
            vmin = -5.0 * vh
            vmax = 5.0 * vh
            vtemporal = vmin + (vmax - vmin)*(sp.random.random())
            f = 0.5 * (sp.exp(-(vtemporal - vh)*(vtemporal - vh) / 2.0) + sp.exp(-(vtemporal + vh)*(vtemporal + vh) / 2.0))
            xtemporal = fmax * (sp.random.random())
            while xtemporal>f:
                fmax = 0.5 * (1. + sp.exp(-2. * vh * vh))
                vmin = -5.0 * vh
                vmax = 5.0 * vh
                vtemporal = vmin + (vmax - vmin)*(sp.random.random())
                f = 0.5 * (sp.exp(-(vtemporal - vh)*(vtemporal - vh) / 2.0) + sp.exp(-(vtemporal + vh)*(vtemporal + vh) / 2.0))
                F[i] = f
                xtemporal = fmax * (sp.random.random())
            
            v[i] = vtemporal
  
    elif(tdistribucion == 2 and Bo != 0):
         """Plasma con campos cruzados (E y B)"""
         F = sp.zeros(nparticulas)
        
         for i in range(nparticulas):    
             fmax = 0.5 * (1. + np.exp(-2. * vh * vh))
             vmin = -5.0 * vh
             vmax = 5.0 * vh
             vtemporal = vmin + (vmax - vmin)*(np.random.random())
             f =  0.5*(np.exp(-(vtemporal - vh)*(vtemporal - vh) / 2.0)) 
             xtemporal = fmax * (np.random.random())
             while xtemporal>f:
                fmax = 0.5 * (1. + np.exp(-2. * vh * vh))
                vmin = -5.0 * vh
                vmax = 5.0 * vh
                vtemporal = vmin + (vmax - vmin)*(np.random.random())
                f = 0.5*(np.exp(-(vtemporal - vh)*(vtemporal - vh) / 2.0))
                F[i] = f
                xtemporal = fmax * (np.random.random())
            
             v[i] = vtemporal
        
         vv = v 
         
         # Se obtienen las componentes de la velocidad a partir de la distribucion de velocidades:           
         vx = vv*sp.cos(2*np.pi*x/longitud_malla)
         vy = vv*sp.sin(2*np.pi*x/longitud_malla)
         
    elif(tdistribucion == 2 and Bo == 0):
        print "Ingrese el valor del campo magnetico"
        sys.exit(0)
        
    elif(tdistribucion == 3):
        """Inestabilidad Beam-Plasma: generada a partir de dos flujos de electrones con distinta densidad"""
        v_movidas = vh    
        n_quietas = nparticulas*0.9
        n_movidas = nparticulas*0.1
        v_quietas = - (v_movidas*n_movidas)/(n_quietas-n_movidas) 
        F = sp.zeros(nparticulas)
    
        for i in range(nparticulas):    
            fmax = 0.5 * (1. + np.exp(-2. * v_movidas * v_movidas))
            vmin = -2. * v_movidas
            vmax = 2. * v_movidas
            vtemporal = vmin + (vmax - vmin)*(np.random.random())
            f =  (1-(n_movidas/n_quietas))*np.exp(-(vtemporal - v_quietas)*(vtemporal - v_quietas)/1.) + (n_movidas/n_quietas)*np.exp(-(vtemporal - v_movidas)*(vtemporal - v_movidas)/1.)
            xtemporal = fmax * (np.random.random())
            while xtemporal>f:
                fmax = 0.5 * (1. + np.exp(-2. * v_movidas * v_movidas))
                vmin = -2. * v_movidas
                vmax = 2. * v_movidas
                vtemporal = vmin + (vmax - vmin)*(np.random.random())
                f =  (1-(n_movidas/n_quietas))*np.exp(-(vtemporal - v_quietas)*(vtemporal - v_quietas)/1.) + (n_movidas/n_quietas)*np.exp(-(vtemporal - v_movidas)*(vtemporal - v_movidas)/1.)
                F[i] = f
                xtemporal = fmax * (np.random.random())
            
            v[i] = vtemporal
             
    # Se agrega la perturbacion
    v += vP*sp.sin(2*sp.pi*x/longitud_malla)
    
    return True
  
#==================================================================
#  TERCERA ETAPA: Se chequea si las particulas sobrepasan las C.F.
#================================================================== 
def particula_cf(xtamanio):      
   global x

   #  Condiciones de frontera periodicas (PBC):
   for i in range(nparticulas):
       if ( x[i] < 0.0 ):
         x[i] += xtamanio
       elif ( x[i] >= xtamanio ):
         x[i] -= xtamanio
      
   return True
  
#==================================================================
#   CUARTA ETAPA: Calculo de las densidades 
#==================================================================
def densidad(qe):      
   """El metodo de ponderacion implementado es el CIC"""
   global x,rhoe,rhoi,dx,nparticulas,npuntos_malla,pared_izquierda,pared_derecha
   
   j1 = sp.dtype(sp.int32) # Asegura que la variable permanezca entera
   j2 = sp.dtype(sp.int32)    
   
   # Factor de ponderacion de carga 
   re = qe/dx 
   # Densidad electronica 
   rhoe = sp.zeros(npuntos_malla+1)   
   # Mapa de cargas sobre la malla
   for i in range(nparticulas):
       xa = x[i]/dx # xparticula/dx
       j1 = int(xa) # indices de la malla fija xmalla/dx
       j2 = j1 + 1  # Siguiente punto en la malla
       f2 = xa - j1 # |xmalla - xparticula|/dx
       f1 = 1.0 - f2
       rhoe[j1] = rhoe[j1] + re*f1
       rhoe[j2] = rhoe[j2] + re*f2

   # Condiciones de frontera periodica
   rhoe[0] += rhoe[npuntos_malla]
   rhoe[npuntos_malla] = rhoe[0]
     
   # Se agrega una densidad de iones neutral
   rhoi = rho0

   return True

#==================================================================
#   QUINTA ETAPA: Calculo deL campo electrico
#==================================================================
def campo():          
  global rhoe,rhoi,ex,dx,npuntos_malla,Ex,x,rhot
    
  rhot=rhoe+rhoi  # Densidad de carga neta sobre la malla
  
  # Integrar div.E=rho directamente usando el metodo del trapecio 
  xintegrar = dx * sp.arange(npuntos_malla+1)
  Ex = sp.integrate.cumtrapz(rhot,xintegrar, initial = xintegrar[0])
  edc = sp.sum(Ex)
  
  
	# Campos periodicos:  Quitar componente DC
	#  -- Para consistencia con la conservacion de carga
  Ex[0:npuntos_malla] -= edc/npuntos_malla  
  Ex[npuntos_malla] = Ex[0]
    
  return True

#==================================================================
#   SEXTA ETAPA: Mover particulas
#==================================================================
def movimientoparticulas():  
    global x,y,v,Ex,dt,dx,nparticulas,carga_masa,Bo
    global vx,vy,v_menos,v_prima,v_mas,exi
    """Se interpola el campo Ex desde la malla a la particula"""
    
    if (Bo != 0):
          v_menos = sp.zeros((nparticulas,3))
          v_prima = sp.zeros((nparticulas,3))
          v_mas = sp.zeros((nparticulas,3))
          vz = sp.zeros(nparticulas)
          B = sp.array([0,0,Bo])
          t = carga_masa*B*dt*0.5
          s = 2*t/(1+t**2)
    
    for i in range(nparticulas):
    # Cloud In Cell
      #-----------------------------------
      xa = x[i]/dx
      j1 = int(xa)
      j2 = j1 + 1
      b2 = xa - j1
      b1 = 1.0 - b2
      # Repartir npuntos_malla valores de campo al numero de particulas ingresado
      exi = b1*Ex[j1] + b2*Ex[j2] 
      #-----------------------------------

      if (Bo == 0):        
          # Actualizar velocidades/Fuerza de Lorentz en diferencias
          v[i] = v[i] + carga_masa*dt*exi 
          
      else:
          # Algoritmo "Metodo de Boris"
          v_menos[i,0] = vx[i] + carga_masa*dt*exi*0.5
          v_menos[i,1] = vy[i]
          v_menos[i,2] = vz[i]
  
          v_menos_cruz_t = sp.cross(v_menos[i,:],t)
      
          v_prima[i,0] = v_menos[i,0] + v_menos_cruz_t[0]
          v_prima[i,1] = v_menos[i,1] + v_menos_cruz_t[1]
          v_prima[i,2] = v_menos[i,2] + v_menos_cruz_t[2]
      
          v_prima_cruz_s = sp.cross(v_prima[i,:],s)
      
          v_mas[i,0] = v_menos[i,0] + v_prima_cruz_s[0]
          v_mas[i,1] = v_menos[i,1] + v_prima_cruz_s[1]
          v_mas[i,2] = v_menos[i,2] + v_prima_cruz_s[2]
      
          vx[i] = v_mas[i,0] + carga_masa*dt*exi*0.5
          vy[i] = v_mas[i,1]
          vz[i] = v_mas[i,2]
     
    #  Actualizar posiciones (2do paso del Leap Frog) 
    if(Bo == 0):
        x += dt*v  
    
    if(Bo != 0):
        x += dt*vx
        y += dt*vy 
       
    return True
#==================================================================
# GRAFICAS DE DIAGNOSTICO
#==================================================================
def diagnosticos():
    """Funcion que genera los graficos de las distintas
       cantidades fisicas en un paso de tiempo determinado
       por el usuario"""      
    global rhoe,Ex,npuntos_malla,itiempo,longitud_malla,rho0,aP,v1,v2,F
    global EnergiaK, EnergiaP, EnergiaT, emax
    global iout,igrafica,ifase,ivdist, distribucion
    global Archivos_Densidades, Archivos_Campo, Archivos_Efase, Archivos_Fdistribucion
    
    # Se crea el eje para graficar las cantidades fisicas involucradas:
    xgrafica = dx * sp.arange(npuntos_malla+1)
    
    if (itiempo == 0): 
        plt.figure('Cantidades')
        plt.clf()
        
    if (igrafica > 0):
        # Se grafica cada paso dado por el contador igrafica:
        if (sp.fmod(itiempo,igrafica) == 0): 
            # Densidad total
            plt.figure(1)
            if (itiempo >0 ): plt.cla()
            plt.plot(xgrafica, -(rhoe+rho0), 'r', label='Densidad')
            plt.xlabel('x')
            plt.xlim(0,longitud_malla)
            plt.ylim(-1.5,1.5)
            plt.legend(loc=1)
            # Se imprimen y se guardan las imagenes de acuerdo a iout:
            plt.pause(0.0001)
            plt.draw()
            filename = '%0*d_densidad'%(5, itiempo)
            Archivos_Densidades[itiempo] = filename
            if (iout > 0):
                if (sp.fmod(itiempo,iout) == 0): 
                    plt.savefig(filename+'.png',dpi=720)
                                                                 
            # Campo electrico
            plt.figure(2)
            if (itiempo >0 ): plt.cla()
            plt.plot(xgrafica, Ex, 'b' , label = 'Ex')
            plt.xlabel('x', fontsize = 18)
            plt.ylabel('Ex', fontsize = 18)
            plt.xticks(np.linspace(0,16,4), fontsize = 18)
            plt.yticks(np.linspace(-0.0010,0.0010,5), fontsize = 18)
            plt.xlim(0,longitud_malla)
            plt.ylim(-0.0015,0.0015)
            plt.legend(loc = 1)
            # Se imprimen y se guardan las imagenes de acuerdo a iout:
            plt.pause(0.0001)
            plt.draw()
            filename = '%0*d_campoelectrico'%(5, itiempo)
            Archivos_Campo[itiempo] = filename
            if (iout > 0):
                if (sp.fmod(itiempo,iout) == 0): 
                    plt.savefig(filename+'.png',dpi=720)
                            
            if (ifase > 0):
              if (sp.fmod(itiempo,ifase) == 0):  
                # Se grafica el espacio de fase en el paso dado por el contador ifase:
                plt.figure(3)
                if (itiempo >0 ): plt.cla()
                v1 = sp.zeros(nparticulas)
                v2 = sp.zeros(nparticulas)
                x1 = sp.zeros(nparticulas)
                x2 = sp.zeros(nparticulas)
                for i in range(nparticulas):
                    if (v[i-1]>v[i]):
                        v1[i]=v[i]
                        x1[i]=x[i]
                    elif(v[i-1]<v[i]):
                        v2[i]=v[i]
                        x2[i]=x[i]  
                if(distribucion == 0):
                    plt.scatter(x,v,marker='.',s=0.1,color='black')  
                elif(distribucion == 1 or distribucion == 2):
                    plt.scatter(x1,v1,marker='.',s=0.1,color='red')        
                    plt.scatter(x2,v2,marker='.',s=0.1,color='blue')
                    plt.xticks(np.linspace(0,100,6), fontsize = 18)
                    plt.yticks(np.linspace(-8,8,5), fontsize = 18)
                    plt.xlabel('x', fontsize = 18)
                    plt.ylabel('v', fontsize = 18)
                elif(distribucion == 3):
                    plt.scatter(x,v,marker='.',s=0.5,color='black')
                    plt.xticks(np.linspace(0,50,6), fontsize = 18)
                    plt.yticks(np.linspace(-4,8,7), fontsize = 18)
                    plt.xlabel('x', fontsize = 18)
                    plt.ylabel('v', fontsize = 18)
                plt.xlim(0,longitud_malla)
                plt.ylim(-4,8)

                # Se imprimen y se guardan las imagenes de acuerdo a iout:
                plt.pause(0.0001)
                plt.draw()
                filename = '%0*d_espaciofase'%(5, itiempo)
                Archivos_Efase[itiempo] = filename
                if (iout > 0):
                    if (sp.fmod(itiempo,iout) == 0):  
                        plt.savefig(filename+'.png',dpi=240)
                                        
            if (ivdist > 0):
              if (sp.fmod(itiempo,ivdist)==0):
                plt.figure(4)
                if (itiempo >0 ): plt.cla()                
                plt.scatter(v,F,marker = '.' , s=0.1, color ='green')
                plt.xlim(-5*vh,5*vh)
                plt.ylim(0,1.0)
                plt.xlabel('v')
                plt.ylabel('f(v)')
                #fn_vdist = 'vdist_%0*d'%(5, itiempo)
                # Se imprimen y se guardan las imagenes de acuerdo a iout:
                plt.pause(0.0001)
                plt.draw()
                filename = '%0*d_fdistribucion'%(5, itiempo)
                Archivos_Fdistribucion[itiempo] = filename
                if (iout > 0):
                    if (sp.fmod(itiempo,iout) == 0):  
                        plt.savefig(filename+'.png',dpi=720)
                     #Se escriben los datos de la distribucion en un archivo:
#                    sp.savetxt(fn_vdist, sp.column_stack((v,F)),fmt=('%1.4e','%1.4e'))   
                        
            if (Bo !=0):
                if (igrafica > 0):
                    if (sp.fmod(itiempo,igrafica)==0):
                        fig = plt.figure(5)
                        grafico_tridi = fig.add_subplot(111,projection='3d')
                        if (itiempo >0 ): plt.cla()
                        grafico_tridi.scatter(x,y,z, marker='.',s=0.1, color = "blue")
                        grafico_tridi.set_xlabel('x')
                        grafico_tridi.set_ylabel('y')
                        grafico_tridi.set_zlabel('z')
                        # Se imprimen y se guardan las imagenes de acuerdo a iout:
                        plt.pause(0.0001)
                        plt.draw()
                        filename = '%0*d_trayectoria'%(5, itiempo)
                        Archivos_Trayectorias[itiempo] = filename
                        if (iout > 0):
                            if (sp.fmod(itiempo,iout) == 0):  
                                plt.savefig(filename+'.png',dpi=720)
                                                         
    # Energia cinetica:
    v2 = v**2
    EnergiaK[itiempo] = 0.5*masa*sum(v2)
  
    # Energia potencial:
    e2 = Ex**2
    EnergiaP[itiempo] = 0.5*dx*sum(e2)
    emax = max(Ex) # Campo maximo para analisis de inestabilidad
 
    # Energia total: 
    EnergiaT[itiempo] = EnergiaP[itiempo] + EnergiaK[itiempo]
    
    return True
    
#==================================================================
# GRAFICAS DE EVOLUCION TEMPORAL DE LA ENERGIA
#==================================================================
def historial():
  """Funcion que permite graficar la energia a lo largo del tiempo especificiado por el usuario"""
  global EnergiaK, EnergiaP, EnergiaT
  
  t = dt*np.arange(npasos_temporales+1)
  plt.figure('Energias del sistema')
  plt.title('Energies')
  plt.plot(t, EnergiaP, 'b', label='Potential')
  plt.plot(t, EnergiaK, 'r', label='Kinetic')
  plt.plot(t, EnergiaT, 'black', label='Total')
  plt.xlabel('t', fontsize = 18)
  plt.xticks(np.linspace(0,14,6), fontsize = 18)
  plt.yticks(np.linspace(0,35e-7,6), fontsize = 18)
  plt.ylim(0,40e-7)
  plt.xlim(0,14)
  plt.legend(loc=1)
  plt.ticklabel_format(style = 'sci', axis = 'y', scilimits = (0,0))
  plt.figure('Potential Energy')
  plt.plot(t, EnergiaP, 'b')
  plt.xlabel('t', fontsize = 18)
  plt.ylabel('Ex Energy', fontsize = 18)
  plt.xticks(np.linspace(0,100,11), fontsize = 18)
  plt.yticks(np.linspace(0,16,8), fontsize = 18)
  plt.xlim(0,100)
  plt.ylim(0,25)
  if  os.path.exists("Energias") and\
  os.path.isfile("Energias/Energias.png")==\
  True:
    os.remove("Energias/Energias.png")            
    plt.savefig('Energias.png',dpi=720)
    shutil.move('Energias.png',"Energias")
    os.remove("Energias/energies.out")
    # Escribe y guarda el archivo con los valores de la energia en el tiempo:
    sp.savetxt('energies.out', sp.column_stack((t,EnergiaP,EnergiaK,EnergiaT)),fmt=('%1.4e','%1.4e','%1.4e','%1.4e'))   
    shutil.move('energies.out',"Energias")     
            
  else:
    os.mkdir("Energias")
    plt.savefig('Energias.png',dpi=720)
    shutil.move('Energias.png',"Energias")       
    # Escribe y guarda el archivo con los valores de la energia en el tiempo:
    sp.savetxt('energies.out', sp.column_stack((t,EnergiaP,EnergiaK,EnergiaT)),fmt=('%1.4e','%1.4e','%1.4e','%1.4e'))   
    shutil.move('energies.out',"Energias")
 
#==================================================================
#  GRAFICA RELACION DE DISPERSION
#==================================================================
def relacion_dispersion(distribucion):
    global npuntos_malla,longitud_malla,dt,npasos_temporales,vh,dx,rhot,Ex 
    global E_acumulador

    # Frecuencia del plasma
    omegap = 1
    # Vector de frecuencia del plasma
    omegapp = omegap*sp.ones(nparticulas)
    
    # Se calcula la frecuencia angular y espacial minima y maxima (Ver codigo KEMPO1):
    omega_min = 2*sp.pi/(dt)/2/(npasos_temporales/2)
    omega_max = omega_min*(npasos_temporales/2)
    k_min = 2*sp.pi/(npuntos_malla)
    k_max = k_min*((npuntos_malla/2)-1)
    
    # Se crean los vectores de frecuencias espacial y angular teoricas y simuladas:
    k_t =sp.linspace(0,k_max,nparticulas)
    k_simulada = sp.linspace(-k_max,k_max,nextpow2(npuntos_malla))
    omega_t = sp.linspace(0,longitud_malla,npuntos_malla+1)
    omega_simulada = sp.linspace(-omega_max,omega_max,nextpow2(npuntos_malla))
    
    # Se genera una matriz de frecuencias angular y espacial:
    K, W = sp.meshgrid(k_simulada,omega_simulada)    
    
    # Se muestrea la matriz espacio-temporal:
    E_acumulador_muestreado = E_acumulador[0:npuntos_malla:1,0:npasos_temporales:(5*npasos_temporales/npasos_temporales)]    
    
    # Se efectua la FFT sobre la matriz espacio temporal muestreada, luego el 
    # valor absoluto de dicha matriz y el corrimiento al cero de las frecuencias:
    E_wk = fft2(E_acumulador_muestreado,(nextpow2(npuntos_malla),nextpow2(npuntos_malla)))/longitud_malla
    E_wk_absoluto = abs(E_wk)
    E_wk_shift = shift(E_wk_absoluto)    
    
    if(distribucion==1):
                
       # Relacion de dispersion teorica inestabilidad Two-Stream:
       omega_t = -1j*(sp.sqrt((vh*k_t)**2 + omegap**2 - omegap*sp.sqrt(4*(k_t*vh)**2 + omegap**2)))
       # Se grafica la relacion de dispersion teorica y la simulada:
       plt.figure('Relacion dispersion (Two-Stream)')
       plt.cla()
       plt.plot(k_t,omega_t,'k',label = 'Relacion de dispersion teorica')
       plt.plot(k_t,omegapp,'k',label = '$\omega_{p}$')
       plt.xlabel('k', fontsize = 18)
       plt.ylabel('$\omega$', fontsize = 18)
       plt.xticks(np.linspace(0,0.7,6), fontsize = 18)
       plt.yticks(np.linspace(0,1,5), fontsize = 18)
       
       # Se grafica la relacion de dispersion simulada:
       plt.contourf(K,W,E_wk_shift, 8, alpha=.75, cmap='jet')
       plt.xlim(0,0.7)
       plt.ylim(0.0,1.1)
       
       if os.path.exists("RelacionesDeDispersion") and\
       os.path.isfile("RelacionesDeDispersion/relaciondispersiontwostream.png")==\
        True:
            os.remove("RelacionesDeDispersion/relaciondispersiontwostream.png")            
            plt.savefig('relaciondispersiontwostream.png',dpi=360)
            shutil.move('relaciondispersiontwostream.png',"RelacionesDeDispersion")
            
       elif os.path.exists("RelacionesDeDispersion") and\
       os.path.isfile("RelacionesDeDispersion/relaciondispersionplasmafrio.png")==\
       True:
            plt.savefig('relaciondispersiontwostream.png',dpi=360)
            shutil.move('relaciondispersiontwostream.png',"RelacionesDeDispersion")       
            
       else:
            os.mkdir("RelacionesDeDispersion")
            plt.savefig('relaciondispersiontwostream.png',dpi=360)
            shutil.move('relaciondispersiontwostream.png',"RelacionesDeDispersion")       
        
    if(distribucion==0):
        
        # Relacion de dispersion teorica para el plasma frio
        plt.figure('Relacion dispersion (Plasma frio)')
        plt.cla()
        plt.plot(k_t,omegapp,'k',label = '$\omega_{p}$')
        plt.xlabel('k', fontsize = 18)
        plt.ylabel('$\omega$', fontsize = 18)
        plt.xticks(np.linspace(0,2,8), fontsize = 18)
        plt.yticks(np.linspace(0,1,5), fontsize = 18)
          
        # Relacion de dispersion "simulada" para el plasma frio
        plt.contourf(K,W,E_wk_shift, 8, alpha=.75, cmap='jet')
        plt.xlim(0,1.2)
        plt.ylim(0.25,1.1)
        if os.path.exists("RelacionesDeDispersion") and\
        os.path.isfile("RelacionesDeDispersion/relaciondispersionplasmafrio.png")==\
        True:
            os.remove("RelacionesDeDispersion/relaciondispersionplasmafrio.png")            
            plt.savefig('relaciondispersionplasmafrio.png',dpi=360)
            shutil.move('relaciondispersionplasmafrio.png',"RelacionesDeDispersion")
            
        elif os.path.exists("RelacionesDeDispersion") and\
        os.path.isfile("RelacionesDeDispersion/relaciondispersiontwostream.png")==\
        True:
            plt.savefig('relaciondispersionplasmafrio.png',dpi=360)
            shutil.move('relaciondispersionplasmafrio.png',"RelacionesDeDispersion")       
            
        else:
            os.mkdir("RelacionesDeDispersion")
            plt.savefig('relaciondispersionplasmafrio.png',dpi=360)
            shutil.move('relaciondispersionplasmafrio.png',"RelacionesDeDispersion")       
                    
    if(distribucion == 3):
        
        # Relacion de dispersion teorica para la beam-plasma instability
        plt.figure('Relacion dispersion (Beam-Plasma Instability)')

        # Relacion de dispersion "simulada" para Beam-Plasma Instability:
        omegappp = omegapp*0.9
        omegappb = omegapp*0.1
        omega_t = vh*k_t - (omegap*0.1)
        omega_t_dos = vh*k_t + (omegap*0.1)
        plt.contourf(K,W,E_wk_shift, 8, alpha=.75, cmap='jet')
        plt.plot(k_t,omegappp,'k',label = '$\omega_{pp}$')
        plt.plot(k_t,omegappb,'k',label = '$\omega_{pb}$')
        plt.plot(k_t,-omegappb,'k',label = '$-\omega_{pb}$')
        plt.plot(k_t,omega_t,'r')
        plt.plot(k_t,omega_t_dos,'r')
        plt.xlim(0,0.4)
        plt.ylim(0,1.1)
        plt.grid('on')
               
    return True
#==================================================================
#  CODIGO PARA MANIPULACION DE ARCHIVOS Y CARPETAS
#==================================================================
def gestion_archivos():
    
    if  os.path.exists("Densidades") == True: 
        shutil.rmtree("Densidades")
        os.mkdir("Densidades")
        for i in range(npasos_temporales+1):
            shutil.move(Archivos_Densidades[i]+'.png',"Densidades")   
    else:
        os.mkdir("Densidades")
        for i in range(npasos_temporales+1):
            shutil.move(Archivos_Densidades[i]+'.png',"Densidades")           

    if  os.path.exists("Campo") == True: 
        shutil.rmtree("Campo")
        os.mkdir("Campo")
        for i in range(npasos_temporales+1):
            shutil.move(Archivos_Campo[i]+'.png',"Campo")   
    else:
        os.mkdir("Campo")
        for i in range(npasos_temporales+1):
            shutil.move(Archivos_Campo[i]+'.png',"Campo")                  

    if  os.path.exists("EspacioDeFase") == True: 
        shutil.rmtree("EspacioDeFase")
        os.mkdir("EspacioDeFase")
        for i in range(npasos_temporales+1):
            shutil.move(Archivos_Efase[i]+'.png',"EspacioDeFase")   
    else:
        os.mkdir("EspacioDeFase")
        for i in range(npasos_temporales+1):
            shutil.move(Archivos_Efase[i]+'.png',"EspacioDeFase") 
            

    if  os.path.exists("FuncionesDeDistribucion") == True: 
        shutil.rmtree("FuncionesDeDistribucion")
        os.mkdir("FuncionesDeDistribucion")
        for i in range(npasos_temporales+1):
            shutil.move(Archivos_Fdistribucion[i]+'.png',"FuncionesDeDistribucion")   
    else:
        os.mkdir("FuncionesDeDistribucion")
        for i in range(npasos_temporales+1):
            shutil.move(Archivos_Fdistribucion[i]+'.png',"FuncionesDeDistribucion") 

    if  os.path.exists("Trayectorias") == True: 
        shutil.rmtree("Trayectorias")
        os.mkdir("Trayectorias")
        for i in range(npasos_temporales+1):
            shutil.move(Archivos_Trayectorias[i]+'.png',"Trayectorias")   
    else:
        os.mkdir("Trayectorias")
        for i in range(npasos_temporales+1):
            shutil.move(Archivos_Trayectorias[i]+'.png',"Trayectorias") 

#==================================================================
#  FUNCION AUXILIAR
#==================================================================           
def nextpow2(longitud_malla):
    """Siguiente potencia de dos, necesaria para graficar el eje de frecuencias"""
    n = 1
    while n < longitud_malla: n *= 2
    return n
#==================================================================
#  PROGRAMA PRINCIPAL: aqui se definen las variables principales y las configuraciones por defecto
#==================================================================
# Numero de particulas:
nparticulas = 1000
# Puntos de malla:
npuntos_malla = 100
# Pasos temporales:         
npasos_temporales = 100
# Tamanio de la caja computacional:
longitud_malla = 10.
# Magnitud del campo magnetico uniforme:
Bo = 10.

# Arreglo de particulas
x = sp.zeros(nparticulas)  
y = sp.zeros(nparticulas)
x_aux = np.empty((npasos_temporales+1,nparticulas))
y_aux = np.empty((npasos_temporales+1,nparticulas))
v = sp.zeros(nparticulas)     
vx = sp.zeros(nparticulas)
vy = sp.zeros(nparticulas)

# Arreglo de mallas
rhoe = sp.zeros(npuntos_malla+1)        # Densidad electronica 
rhoi = sp.zeros(npuntos_malla+1)        # Densidad ionica 
rhot = sp.zeros(npuntos_malla+1)        # Densidad total
Ex   = sp.zeros(npuntos_malla+1)        # Campo electrico  

# Historial de energia
EnergiaK   = sp.zeros(npasos_temporales+1)
EnergiaP   = sp.zeros(npasos_temporales+1)
EnergiaT   = sp.zeros(npasos_temporales+1)

# Matriz espacio-temporal del campo E:
E_acumulador = sp.empty((npuntos_malla+1,npasos_temporales+1))
# Numero de onda y frecuencia relacion de dispersion
k_s = sp.linspace(0,longitud_malla,npuntos_malla+1)          # Simulada
omega_t = sp.linspace(0,longitud_malla,npuntos_malla+1)      # Teorica
omega_s = sp.linspace(0,longitud_malla,npuntos_malla+1)      # Simulada

# Arreglos para la gestion de archivos
Archivos_Densidades = range(0,npasos_temporales+1)
Archivos_Campo = range(0,npasos_temporales+1)
Archivos_Efase = range(0,npasos_temporales+1)
Archivos_Fdistribucion = range(0,npasos_temporales+1)
Archivos_Trayectorias = range(0,npasos_temporales +1)
#==================================================================
plasma_li = 0.              # Limite izquierdo del plasma
plasma_lf = longitud_malla  # Limite derecho del plasma
dx = longitud_malla/npuntos_malla
dt = 0.1                    # Paso de tiempo normalizado  
carga_masa=-1.0             # Relacion carga-masa
rho0 = 1.0                  # Densidad de fondo de los electrones
aP = 0.0                    # Amplitud de la perturbacion de la posicion
vP = 0.0                    # Amplitud de la perturbacion de la velocidad
vh = 1.0                    # Velocidad maxima para la grafica de f(v)
pared_izquierda = 0.
pared_derecha = 1.
profile = 1   	            # Switch del perfil de densidad
distribucion = 2            # Cambiar la distribucion de velocidad: 0 = plasma frio, 1 = 2-stream 	2 = Flujo bajo la accion de un campo B uniforme  3 = Beam-Plasma Instability
ihist = 5                   # Frecuencia de salida de tiempo-historia
igrafica = int(sp.pi/dt/16) # Frecuencia de las capturas de pantalla de las graficas
ifase = igrafica
ivdist = igrafica
iout = igrafica*1           # Frecuencia de guardado de los archivos de las graficas
itiempo = 0                 # Inicializacion del contador de tiempo 
#==================================================================
#  Validacion de parametros de entrada
#==================================================================
salto = int (npasos_temporales / dt) / 10;

if nparticulas < 1 or npuntos_malla < 2 or longitud_malla <= 0.0\
    or vh <= 0.0 or dt <= 0.0  or npasos_temporales <= 0. or salto < 1:
        print 'Error - revisar los argumentos ingresados'
        sys.exit(0)
#==================================================================
#  Configuracion inicial de la distribucion de particulas y campos:
#==================================================================
cargarposicion()                          # Se cargan las particulas sobre la malla
cargarvelocidad(distribucion,vh)          # Se cargan las velocidad de las particulas
x += 0.5*dt*v                             # Hace el primer avanzo a medio intervalo para la posicion (LEAP FROG)
particula_cf(longitud_malla)              # Se verifica que las particulas esten dentro de la frontera
densidad(carga)                           # Se calcula la densidad inicial
campo()                                   # Se calcula el campo inicial
#diagnosticos()                            # Se generan las primeras graficas de diagnostico
print 'Configuracion inicial completada...'

#==================================================================
#  Ciclo principal (PIC):
#==================================================================
for itiempo in range(1,npasos_temporales+1):
    print 'Paso en el tiempo ',itiempo
    movimientoparticulas()			          # "Empuja" las particulas
    for i in range(0,nparticulas):
        x_aux[itiempo,i] = x[i]
        y_aux[itiempo,i] = y[i]
    particula_cf(longitud_malla)  
    densidad(carga)	    
    campo()		               
    for i in range(1,npuntos_malla+1):
        E_acumulador[i,itiempo] = Ex[i]       # Matriz espacio-temporal del campo electrico               
    #diagnosticos()	                          # Se generan capturas de cada diagnostico
    
print 'Fin del ciclo'
  
#historial()                                   # Se muestra la energia a lo largo del tiempo especificado
#relacion_dispersion(distribucion)             # Se grafica la relacion de dispersion para la Two-Stream
#gestion_archivos()
print 'Fin'

plt.figure('Trayectoria_1particula')
plt.plot(x_aux[:,200],y_aux[:,200])
plt.plot(x_aux[:,400],y_aux[:,400])
plt.plot(x_aux[:,300],y_aux[:,300])
plt.xticks(np.linspace(0,4.5,10), fontsize = 18)
plt.yticks(np.linspace(0,1.5,10), fontsize = 18)
plt.xlabel('x', fontsize = 18)
plt.ylabel('y', fontsize = 18)
