# Ciclo de entrenamiento del notebook 01 parametrizado por "modo", para poder comparar:
#   "notebook"   : exactamente lo que hace el notebook 01 (Adam 1e-2, UN paso full-batch por
#                  iteracion, seleccion elite/super por percentil sin tope).
#   "wagner_fit" : lo que hace Wagner con model.fit (SGD 1e-4, minilotes de 32, 1 epoca) y su
#                  seleccion con tope (n_sessions*(100-pct)/100 sesiones como maximo en el empate).
# Ademas registra cosas que el notebook no imprime: cuantas sesiones entran a elite/super y
# cuantas son grafos distintos (sirve para ver el colapso de la politica).
import time
from nb01_core import *


def seleccionar_wagner(REC, pct, n_ses):
    # umbral = percentil; las sesiones estrictamente por encima entran siempre, las que
    # empatan justo en el umbral entran hasta completar el cupo (asi lo hace select_elites)
    thr = np.percentile(REC, pct)
    cupo = n_ses * (100.0 - pct) / 100.0
    mask = np.zeros(len(REC), dtype=bool)
    for i in range(len(REC)):
        if REC[i] >= thr - 1e-7:
            if cupo > 0 or REC[i] >= thr + 1e-7:
                mask[i] = True
            cupo -= 1
    return mask


def entrenar_variante(iteraciones, modo="notebook", n_ses=n_sessions, carpeta="datos",
                      tiempo_max=None, log=print, guardar_cada=20):
    os.makedirs(carpeta, exist_ok=True)
    t0 = time.time()
    modelo = ModeloGrafo(observation_space).to(device)
    if modo == "notebook":
        opt = torch.optim.Adam(modelo.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    elif modo == "wagner_fit":
        opt = torch.optim.SGD(modelo.parameters(), lr=1e-4)      # Wagner: SGD(learning_rate=0.0001)
    else:
        raise ValueError(modo)
    bce = nn.BCELoss()

    sup_est = np.zeros((0, observation_space, len_game), dtype=np.float32)
    sup_acc = np.zeros((0, len_game), dtype=np.int8)
    sup_rec = np.zeros(0)
    historial = []          # una fila por iteracion
    linea_tiempo = []       # mejor palabra cada guardar_cada iteraciones (para la figura tipo Wagner)

    for it in range(iteraciones):
        if tiempo_max is not None and time.time() - t0 > tiempo_max:
            log(f"tope de tiempo ({tiempo_max:.0f}s) alcanzado en iter {it}")
            break
        est, acc = generar_sesiones(modelo, n_ses)
        rec = recompensas(acc)
        EST = np.concatenate([est, sup_est])
        ACC = np.concatenate([acc, sup_acc])
        REC = np.concatenate([rec, sup_rec])

        if modo == "notebook":
            m_elite = REC >= np.percentile(REC, PCT_ELITE)
            m_super = REC >= np.percentile(REC, PCT_SUPER)
        else:
            m_elite = seleccionar_wagner(REC, PCT_ELITE, n_ses)
            m_super = seleccionar_wagner(REC, PCT_SUPER, n_ses)

        X = EST[m_elite].transpose(0, 2, 1).reshape(-1, observation_space)
        y = ACC[m_elite].reshape(-1).astype(np.float32)
        if modo == "notebook":
            opt.zero_grad()
            pred = modelo(torch.from_numpy(X).to(device))
            loss = bce(pred, torch.from_numpy(y).to(device).unsqueeze(1))
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), GRAD_CLIP)
            opt.step()
            loss_val = loss.item()
        else:                                   # = model.fit(X, y): 1 epoca, batch 32, barajado
            idx = np.random.permutation(len(X))
            tot = 0.0
            for b in range(0, len(X), 32):
                j = idx[b:b + 32]
                opt.zero_grad()
                pred = modelo(torch.from_numpy(X[j]).to(device))
                loss = bce(pred, torch.from_numpy(y[j]).to(device).unsqueeze(1))
                loss.backward()
                opt.step()
                tot += loss.item() * len(j)
            loss_val = tot / len(X)

        sup_est, sup_acc, sup_rec = EST[m_super], ACC[m_super], REC[m_super]
        k = int(np.argmax(REC))
        fila = dict(iter=it, mejor=float(REC[k]), top100=float(np.sort(REC)[-100:].mean()), loss=loss_val,
                    n_elite=int(m_elite.sum()), n_super=int(len(sup_rec)),
                    super_distintos=len(set(map(bytes, sup_acc))), segundos=round(time.time() - t0, 1))
        historial.append(fila)
        log(f"iter {it:4d} | mejor {fila['mejor']:+.4f} | top100 {fila['top100']:+.4f} | loss {loss_val:.4f} | "
            f"elite {fila['n_elite']} super {fila['n_super']} (distintos {fila['super_distintos']})")

        if it % guardar_cada == 0 or REC[k] > 0:
            linea_tiempo.append(dict(iter=it, score=float(REC[k]), palabra=ACC[k].astype(int).tolist()))
            with open(os.path.join(carpeta, f"super_iter_{it}.pkl"), "wb") as f:
                pickle.dump({"acciones": sup_acc, "recompensas": sup_rec}, f)
            torch.save(modelo.state_dict(), os.path.join(carpeta, "modelo.pt"))
        if REC[k] > 0:
            np.save(os.path.join(carpeta, "contraejemplo.npy"), ACC[k])
            log(f"CONTRAEJEMPLO encontrado, score = {REC[k]}")
            break

    ultimo = dict(iter=it, score=float(REC[k]), palabra=ACC[k].astype(int).tolist())
    mejor_global = max(linea_tiempo + [ultimo], key=lambda d: d["score"])
    return dict(modelo=modelo, historial=historial, linea_tiempo=linea_tiempo, mejor=mejor_global,
                segundos=time.time() - t0)
