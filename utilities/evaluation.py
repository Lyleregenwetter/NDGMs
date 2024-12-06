from pymoo.indicators.hv import HV
from pymoo.indicators.gd import GD
import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
import sklearn
from tqdm import trange
import eval_prd

from scipy.spatial.distance import cdist

def Hypervolume_wrapper(hv_ref="auto"):
    def Hypervolume(x_eval, y_eval, x_data, y_data, n_data, scorebars, hv_ref=hv_ref):
        y_eval = np.array(y_eval)
        if scorebars:
            print("Calculating Hypervolume")
        if hv_ref=="auto":
            hv_ref = np.quantile(y_eval, 0.99, axis=0)
            print("Warning: no reference point provided!")
        hv = HV(ref_point=hv_ref)
        hvol = hv(y_eval)
        return None, hvol
    return Hypervolume

def Generational_distance_wrapper(pf):
    def Generational_distance(x_eval, y_eval, x_data, y_data, n_data, scorebars, pf=pf):
        y_eval = np.array(y_eval)
        if scorebars:
            print("Calculating Generational Distance")
        gd = GD(pf)
        hvol = gd(y_eval)
        return None, hvol
    return Generational_distance

def get_perc_band(value, data, band):
    perc = sum(data < value) / len(data)
    if perc < band/2:
        lower = 0
        upper = band
    elif perc > 1-band/2:
        lower = 1-band
        upper = 1
    else:
        lower = perc-band/2
        upper = perc+band/2
    lb = np.quantile(data, lower)
    ub = np.quantile(data, upper)
    mask = np.logical_and(data>=lb, data<=ub)
    return mask

def calc_distance(X, Y, distance="Euclidean"):
    if distance=="Euclidean":
        return L2_vectorized(X,Y)
    else:
        raise Exception("Unknown distance metric specified")
        
def L2_vectorized(X, Y):
    return cdist(X, Y, 'euclidean')

def signed_distance_to_boundary_wrapper(direction, ref, p_, method="linear"):
    def signed_distance_to_boundary(x_eval, y_eval, x_data, y_data, n_data, scorebars, direction=direction, ref=ref, p_=p_, method=method):
        if method=="linear":
            diff = np.subtract(y_eval,ref)
        elif method=="log":
            diff = np.log(np.divide(y_eval, ref))
        else:
            raise Exception("Unknown method, expected linear or log")
            
        if direction=="maximize":
            pass
        elif direction=="minimize":
            diff=-diff
        else:
            raise Exception("Unknown optimization direction, expected maximize or minimize")
        diff_sc = np.multiply(diff, p_)
        diff_clip = np.minimum(diff_sc, np.zeros_like(diff_sc))
        zeros = np.expand_dims(np.zeros(np.shape(diff)[1]), axis=0)
        dists_clip = L2_vectorized(diff_clip, zeros)
        dists = diff_sc.min(axis=1, keepdims=True)
        dists_mask = np.all(diff > 0, axis=1)
        final_scores = np.multiply(dists_mask, dists)-dists_clip
        return final_scores, np.mean(final_scores)
    return signed_distance_to_boundary
        
def gen_gen_distance_wrapper(flag, reduction="min", distance="Euclidean"):
    def gen_gen_distance(x_eval, y_eval, x_data, y_data, n_data, scorebars, flag=flag):
        if scorebars:
            print("Calculating Gen-Gen Distance")
        if flag == "x":
            x = x_eval
        elif flag == "y":
            x = y_eval
        elif flag == "all":
            x = pd.concat([x_eval, y_eval], axis=0)
        else:
            raise Exception("Unknown flag passed")
        res = calc_distance(x, x, distance)
        np.fill_diagonal(res, np.max(res))
        if reduction == "min":
            scores = np.min(res, axis=1)
        elif reduction == "ave":
            scores = np.mean(res, axis=1)
        else:
            raise Exception("Unknown reduction method")
        return scores, np.mean(scores)
    return gen_gen_distance

def distance_to_centroid_wrapper(flag, distance="Euclidean"):
    def distance_to_centroid(x_eval, y_eval, x_data, y_data, n_data, scorebars, flag=flag):
        if scorebars:
            print("Calculating Distance to Centroid")
        if flag == "x":
            x = x_eval
        elif flag == "y":
            x = y_eval
        elif flag == "all":
            x = pd.concat([x_eval, y_eval], axis=0)
        else:
            raise Exception("Unknown flag passed")
        centroid = np.mean(x, axis=0)
        vec = np.subtract(x, centroid)
        distance = np.linalg.norm(vec, axis=1)
        return distance, np.mean(distance)
    return distance_to_centroid

def DPP_diversity_wrapper(flag, subset_size=500, n_eval=10, norm = False):
    def DPP_diversity(x_eval, y_eval, x_data, y_data, n_data, scorebars, flag=flag, subset_size=subset_size, n_eval=n_eval, norm=norm):
        if scorebars:
            print("Calculating DPP Diversity")
        if flag == "x":
            x = x_eval
        elif flag == "y":
            x = y_eval
        elif flag == "all":
            x = pd.concat([x_eval, y_eval], axis=0)
        else:
            raise Exception("Unknown flag passed")
        all_loss = 0
        for i in range(n_eval):
            idx = np.random.choice(x.shape[0], subset_size, replace=False)
            c = x[idx]
            if norm:
                c = c/np.sqrt(c.shape[1])
            r = np.sum(np.square(c), axis=1, keepdims=True)
            D = r - 2 * np.dot(c, c.T) + r.T
            S = np.exp(-0.5 * np.square(D))
            try:
                eig_val, _ = np.linalg.eigh(S)
            except: 
                eig_val = np.ones(c.shape[0])
            loss = -np.mean(np.log(np.maximum(eig_val, 1e-7)))
            all_loss += loss
        DPP_diversity = all_loss / n_eval
        return None, DPP_diversity
    return DPP_diversity

def gen_data_distance_wrapper(flag, reduction="min", distance="Euclidean"):
    def gen_data_distance(x_eval, y_eval, x_data, y_data, n_data, scorebars, flag = flag):
        if scorebars:
            print("Calculating Gen-Data Distance")
        if flag == "x":
            x = x_eval
            data = x_data
        elif flag == "y":
            x = y_eval
            data = y_data
        elif flag == "all":
            x = pd.concat([x_eval, y_eval], axis=0)
            data = pd.concat([x_data, y_data], axis=0)
        else:
            raise Exception("Unknown flag passed")
        res = calc_distance(x, data, distance)

        if reduction == "min":
            scores = np.min(res, axis=1)
        elif reduction == "ave":
            scores = np.mean(res, axis=1)
        else:
            raise Exception("Unknown reduction method")
        return scores, np.mean(scores)
    return gen_data_distance

def data_gen_distance_wrapper(flag, reduction="min", distance="Euclidean"):
    def data_gen_distance(x_eval, y_eval, x_data, y_data, n_data, scorebars, flag = flag):
        if scorebars:
            print("Calculating Data-Gen Distance")
        if flag == "x":
            x = x_eval
            data = x_data
        elif flag == "y":
            x = y_eval
            data = y_data
        elif flag == "all":
            y_eval = pd.concat([x_eval, y_eval], axis=0)
            data = pd.concat([x_data, y_data], axis=0)
        else:
            raise Exception("Unknown flag passed")
        res = calc_distance(data, x, distance)

        if reduction == "min":
            scores = np.min(res, axis=1)
        elif reduction == "ave":
            scores = np.mean(res, axis=1)
        else:
            raise Exception("Unknown reduction method")
        return None, np.mean(scores)
    return data_gen_distance

def DTAI_wrapper(direction, ref, p_, a_):
    def DTAI(x_eval, y_eval, x_data, y_data, n_data, scorebars, direction=direction, ref=ref, p_=p_, a_=a_, DTAI_EPS=1e-7):
        y_eval = np.maximum(y_eval, DTAI_EPS)
        if scorebars:
            print("Calculating DTAI")

        if direction=="maximize":
            x = y_eval / ref
        elif direction=="minimize":
            x = ref / y_eval
        else:
            raise Exception("Unknown optimization direction, expected maximize or minimize")
        case1 = p_ * x - p_
        p_over_a = p_ / a_
        exponential = np.exp(a_ * (1 - x))
        case2 = p_over_a * (1 - exponential)
        casemask = x > 1
        scores = case2 * casemask + case1 * (1 - casemask)
        scores = np.sum(scores, axis=1)         
        smax = np.sum(p_ / a_)
        smin = -np.sum(p_)

        scores = (scores - smin) / (smax - smin)
        return scores, np.mean(scores)
    return DTAI

def weighted_target_success_rate_wrapper(direction, ref, p_):
    def weighted_target_success_rate(x_eval, y_eval, x_data, y_data, n_data, scorebars, direction=direction, ref=ref, p_=p_):
        if scorebars:
            print("Calculating Weighted Target Success Rate")
        y_eval = np.array(y_eval)
        ref = np.array(ref)
        p_ = np.array(p_)
        if direction == "maximize":
            res = y_eval > ref
        elif direction == "minimize":
            res = y_eval < ref
        else:
            raise Exception("Unknown optimization direction, expected maximize or minimize")
        scores = res.astype(float)
        scaled_scores = np.dot(scores, p_) / np.sum(p_)
        return scaled_scores, np.mean(scaled_scores)
    return weighted_target_success_rate

def gen_neg_distance_wrapper(reduction = "min", distance="Euclidean"):
    def gen_neg_distance(x_eval, y_eval, x_data, y_data, n_data, scorebars):
        if scorebars:
            print("Calculating Gen-Neg Distance")
        res = calc_distance(x_eval, n_data, distance)
        if reduction == "min":
            scores = np.min(res, axis=1)
        elif reduction == "ave":
            scores = np.mean(res, axis=1)
        else:
            raise Exception("Unknown reduction method")
        return scores, np.mean(scores)
    return gen_neg_distance

def MMD_wrapper(flag, sigma=0.02, batch_size=1000, num_iter=10, biased=True):
    def MMD(x_eval, y_eval, x_data, y_data, n_data, scorebars, flag=flag, sigma=sigma, biased=biased):
        if scorebars:
            print("Calculating Maximum Mean Discrepancy")
        if flag == "x":
            x = x_eval
            data = x_data
        elif flag == "y":
            x = y_eval
            data = y_data
        elif flag == "all":
            x = pd.concat([x_eval, y_eval], axis=0)
            data = pd.concat([x_data, y_data], axis=0)
        else:
            raise Exception("Unknown flag passed")
        total = 0
        for _ in range(num_iter):
            if len(x) > batch_size:
                X = x[np.random.randint(x.shape[0], size=batch_size), :]    
            else:
                X = x
            if len(data) > batch_size:
                Y = data[np.random.randint(data.shape[0], size=batch_size), :]
            else:
                Y = data
            gamma = 1 / (2 * sigma**2)
        
            XX = np.dot(X, X.T)
            XY = np.dot(X, Y.T)
            YY = np.dot(Y, Y.T)
        
            X_sqnorms = np.diag(XX)
            Y_sqnorms = np.diag(YY)
        
            K_XY = np.exp(-gamma * (
                    -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
            K_XX = np.exp(-gamma * (
                    -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))
            K_YY = np.exp(-gamma * (
                    -2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
            
            if biased:
                mmd2 = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
            else:
                m = K_XX.shape[0]
                n = K_YY.shape[0]
        
                mmd2 = ((K_XX.sum() - m) / (m * (m - 1))
                    + (K_YY.sum() - n) / (n * (n - 1))
                    - 2 * np.mean(K_XY))
            total += mmd2
        return None, total / num_iter
    return MMD

def F_wrapper(flag, beta=1, num_clusters=200, num_angles=1001, num_runs=5, enforce_balance=False):
    def calc_prd(x_eval, y_eval, x_data, y_data, n_data, scorebars):
        if scorebars:
            print("Calculating F" + str(beta))
        if os.path.isfile(f"temp_eval_recall_{flag}.npy") and os.path.isfile(f"temp_eval_precision_{flag}.npy"):
            recall = np.load(f"temp_eval_recall_{flag}.npy")
            precision = np.load(f"temp_eval_precision_{flag}.npy")
        else:
            if flag == "x":
                x = x_eval
                data = x_data
            elif flag == "y":
                x = y_eval
                data = y_data
            elif flag == "all":
                x = pd.concat([x_eval, y_eval], axis=0)
                data = pd.concat([x_data, y_data], axis=0)
            else:
                raise Exception("Unknown flag passed")
            recall, precision = eval_prd.compute_prd_from_embedding(x, data, num_clusters=num_clusters, num_angles=num_angles, num_runs=num_runs, enforce_balance=enforce_balance)
            np.save(f"temp_eval_recall_{flag}.npy", recall)
            np.save(f"temp_eval_precision_{flag}.npy", precision)
        F = eval_prd._prd_to_f_beta(precision, recall, beta=beta, epsilon=1e-10)
        return None, max(F)
    return calc_prd

def AUC_wrapper(flag, num_clusters=200, num_angles=1001, num_runs=5, enforce_balance=False, plot=False):
    def calc_prd(x_eval, y_eval, x_data, y_data, n_data, scorebars):
        if scorebars:
            print("Calculating AUC")
        if os.path.isfile(f"temp_eval_recall_{flag}.npy") and os.path.isfile(f"temp_eval_precision_{flag}.npy"):
            recall = np.load(f"temp_eval_recall_{flag}.npy")
            precision = np.load(f"temp_eval_precision_{flag}.npy")
        else:
            if flag == "x":
                x = x_eval
                data = x_data
            elif flag == "y":
                x = y_eval
                data = y_data
            elif flag == "all":
                x = pd.concat([x_eval, y_eval], axis=0)
                data = pd.concat([x_data, y_data], axis=0)
            else:
                raise Exception("Unknown flag passed")
            recall, precision = eval_prd.compute_prd_from_embedding(x, data, num_clusters=num_clusters, num_angles=num_angles, num_runs=num_runs, enforce_balance=enforce_balance)
            np.save(f"temp_eval_recall_{flag}.npy", recall)
            np.save(f"temp_eval_precision_{flag}.npy", precision)
        F1 = eval_prd._prd_to_f_beta(precision, recall, beta=1, epsilon=1e-10)
        prd_data = [np.array([precision,recall])]
        if plot:
            eval_prd.plot(prd_data, labels=None, out_path=None,legend_loc='lower left', dpi=300)
        tot = 0
        for i in range(len(precision)-1):
            tot += (precision[i] + precision[i+1]) / 2 * (recall[i+1] - recall[i])
        
        return None, tot
    return calc_prd

def evaluate_validity(x_fake, validityfunction):
    scores = validityfunction(x_fake)
    return scores, np.mean(scores)

def convex_hull_wrapper(flag):
    def convex_hull(x_eval, y_eval, x_data, y_data, n_data, scorebars):
        if scorebars:
            print("Calculating Convex Hull")
        if flag == "x":
            x = x_eval
        elif flag == "y":
            x = y_eval
        elif flag == "all":
            x = pd.concat([x_eval, y_eval], axis=0)
        else:
            raise Exception("Unknown flag passed")
        hull = ConvexHull(x)
        return None, hull.volume
    return convex_hull

def predicted_conditioning_wrapper(reg, cond):
    def predicted_conditioning(x_eval, y_eval, x_data, y_data, n_data, scorebars, reg=reg, cond=cond):
        if scorebars:
            print("Calculating predicted_constraint_satisfaction")
        c_data = y_data
        reg.fit(x_data, c_data)
        res = reg.predict(x_eval)
        cond = np.ones_like(res) * cond
        score = sklearn.metrics.mean_squared_error(res, cond)
        return None, score
    return predicted_conditioning

def predicted_constraint_satisfaction_wrapper(clf):
    def predicted_constraint_satisfaction(x_eval, y_eval, x_data, y_data, n_data, scorebars, clf=clf):
        if scorebars:
            print("Calculating predicted_constraint_satisfaction")
        x_all = np.concatenate([x_data, n_data], axis=0)
        y_all = np.concatenate([np.ones(len(x_data)), np.zeros(len(n_data))], axis=0)
        clf.fit(x_all, y_all)
        res = clf.predict_proba(x_eval)[:, 1]
        return res, np.mean(res)
    return predicted_constraint_satisfaction

def ML_efficacy_wrapper(clf, score):
    def ML_efficacy(x_eval, y_eval, x_data, y_data, n_data, scorebars, clf=clf, score=score):
        if scorebars:
            print("Calculating ML Efficacy")
        clf.fit(x_eval, y_eval)
        preds = clf.predict(x_data)
        res = score(y_data, preds)
        
        return res, np.mean(res)
    return ML_efficacy
