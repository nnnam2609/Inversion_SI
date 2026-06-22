import torch
import os
from scipy.spatial.distance import cdist, pdist, squareform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.colors import COLORS
from sklearn.decomposition import PCA


ARYTENOID_MUSCLE = "arytenoid-muscle"
ARYTENOID_CARTILAGE = "arytenoid-cartilage"
EPIGLOTTIS = "epiglottis"
LOWER_INCISOR = "lower-incisor"
LOWER_LIP = "lower-lip"
PHARYNX = "pharynx"
SOFT_PALATE = "soft-palate"
SOFT_PALATE_MIDLINE = "soft-palate-midline"
THYROID_CARTILAGE = "thyroid-cartilage"
TONGUE = "tongue"
UPPER_INCISOR = "upper-incisor"
UPPER_LIP = "upper-lip"
VOCAL_FOLDS = "vocal-folds"

ART_SLICES = {
    "tongue-tip": (30, 45),
    "tongue-body": (10, 30),
    "tongue-root": (0, 30),
    "upper-incisor": (0, 25),
    "dental": (6, 7),
    "hard-palate": (25, 50),
    "soft-palate": (0, 15),
    "velum": (35, 50),
    "epiglottis": (0, 15),
    "arytenoid-cartilage":(10,40)
}

selected_indices = [0, 9, 14, 19, 24, 29, 34, 44, 49]


def save_incremental_plot(folder, base_filename):
    os.makedirs(folder, exist_ok=True)  # Crée le dossier s'il n'existe pas
    i = 1
    while True:
        filename = os.path.join(folder, f"{base_filename}_{i}.png")
        if not os.path.exists(filename):
            plt.savefig(filename)
            print(f"Figure enregistrée : {filename}")
            break
        i += 1
        

def plot_vt(inputs_dict, frame_name, phoneme,TVs, folder_run, prefix=""):

    name_file = f"{int(frame_name[0])}_S{int(frame_name[1])}_{int(frame_name[2]):04d}"
    image_path = f"/srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/ArtSpeech_Database_2/{int(frame_name[0])}/S{int(frame_name[1])}/NPY_MR_registered/{int(frame_name[2]):04d}.npy"
    image_data = np.load(image_path)
    
    folder_TV = os.path.join(folder_run, f'TV_{prefix}')
    os.makedirs(folder_TV, exist_ok=True)  # Crée le dossier s'il n'existe pas
    
    texts = [
            
            f"PH : {TVs['PH']['value']:.2f}    VEL : {TVs['VEL']['value']:.2f}",
            f"TRCL: {TVs['TRCL']['value']:.2f}",
            f"TRCD : {TVs['TRCD']['value']:.2f}",
            f"TBCD : {TVs['TBCD']['value']:.2f}",
            f"TTCD : {TVs['TTCD']['value']:.2f}",
            
            f"LD : {TVs['LD']['value']:.2f}",
            f"LA : {TVs['LA']['value']:.2f}",
            f"LP : {TVs['LP']['value']:.2f}",
            
            #f"VEL : {TVs['VEL']['value']:.2f}"
            ]
    
    
    
    plt.figure()
    plt.imshow(image_data, cmap='gray')
    for name, articulator in inputs_dict.items():
        # if name not in ["upper-incisor", "pharynx", "vocal-folds"]:
        #     continue
        if isinstance(articulator, list):
            articulator = np.array(articulator)
        if articulator.shape[1] != 2:
            print(f"Attention : {name} n'a pas la forme attendue (N, 2)")
            continue
        # if name =="upper-incisor":
        #     plt.plot(articulator[:, 0], articulator[:, 1],color=COLORS[name])
        else:
            plt.plot(articulator[:, 0], articulator[:, 1], color=COLORS[name])
        
    plt.scatter(*TVs["LP"]["poc_1"], color=COLORS['upper-incisor'],s=10)
    plt.scatter(*TVs["LP"]["poc_2"], color=COLORS['upper-lip'],s=10)    
    plt.scatter(*TVs["LA"]["poc_1"], color=COLORS['lower-lip'],s=10)
    plt.scatter(*TVs["LA"]["poc_2"], color=COLORS['upper-lip'],s=10)
    plt.scatter(*TVs["LD"]["poc_1"], color=COLORS['tongue'],s=10)
    plt.scatter(*TVs["LD"]["poc_2"], color=COLORS['upper-incisor'],s=10)
    
    plt.scatter(*TVs["TTCD"]["poc_1"], color=COLORS['tongue'],s=10)
    plt.scatter(*TVs["TTCD"]["poc_2"], color=COLORS['upper-incisor'],s=10)
    plt.scatter(*TVs["TBCD"]["poc_1"], color=COLORS['tongue'],s=10)
    #plt.scatter(*TVs["TBCD"]["poc_2"], color=COLORS['upper-incisor'],s=10)
    plt.scatter(*TVs["TBCD"]["poc_2"], color=COLORS['soft-palate-midline'],s=10)
    plt.scatter(*TVs["TRCD"]["poc_1"], color=COLORS['tongue'],s=10)
    plt.scatter(*TVs["TRCD"]["poc_2"], color=COLORS['pharynx'],s=10)
    plt.scatter(*TVs["TRCL"]["poc_1"], color=COLORS['tongue'],s=10)
    plt.scatter(*TVs["TRCL"]["poc_2"], color=COLORS['soft-palate-midline'],s=10)
    
    plt.scatter(*TVs["PH"]["poc_1"], color=COLORS['vocal-folds'],s=10)
    plt.scatter(*TVs["PH"]["poc_2"], color=COLORS['upper-incisor'],s=10)
    #plt.scatter(*inputs_dict[UPPER_INCISOR][-1], color=COLORS['upper-incisor'],s=10)
    
    plt.plot([TVs["LP"]["poc_1"][0], TVs["LP"]["poc_2"][0]], [TVs["LP"]["poc_1"][1], TVs["LP"]["poc_2"][1]], '--', color="thistle")#, label='LP')
    plt.plot([TVs["LA"]["poc_1"][0], TVs["LA"]["poc_2"][0]], [TVs["LA"]["poc_1"][1], TVs["LA"]["poc_2"][1]], '--', color="mediumslateblue")#, label='LA')
    plt.plot([TVs["LD"]["poc_1"][0], TVs["LD"]["poc_2"][0]], [TVs["LD"]["poc_1"][1], TVs["LD"]["poc_2"][1]], '--', color="darkolivegreen")#, label='LD')
    
    plt.plot([TVs["TTCD"]["poc_1"][0], TVs["TTCD"]["poc_2"][0]], [TVs["TTCD"]["poc_1"][1], TVs["TTCD"]["poc_2"][1]], '--', color="saddlebrown")#, label='TTCD')
    plt.plot([TVs["TBCD"]["poc_1"][0], TVs["TBCD"]["poc_2"][0]], [TVs["TBCD"]["poc_1"][1], TVs["TBCD"]["poc_2"][1]], '--', color="darkkhaki")#, label='TBCD')
    plt.plot([TVs["TRCD"]["poc_1"][0], TVs["TRCD"]["poc_2"][0]], [TVs["TRCD"]["poc_1"][1], TVs["TRCD"]["poc_2"][1]], '--', color="palegreen")#, label='TRCD')
    plt.plot([TVs["TRCL"]["poc_1"][0], TVs["TRCL"]["poc_2"][0]], [TVs["TRCL"]["poc_1"][1], TVs["TRCL"]["poc_2"][1]], '--', color="violet")#, label='TRCL')
    
    plt.plot([TVs["PH"]["poc_1"][0], TVs["PH"]["poc_2"][0]], [TVs["PH"]["poc_1"][1], TVs["PH"]["poc_2"][1]], '--', color="crimson")#, label='PH')
    #plt.plot([inputs_dict[UPPER_INCISOR][-1,0], TVs["PH"]["poc_2"][0]], [inputs_dict[UPPER_INCISOR][-1,1], TVs["PH"]["poc_2"][1]], '--', color='yellow')#, label='PH')
    
    for i, txt in enumerate(texts):
        plt.text(
            2,                      # x = 2 pixels depuis la gauche
            image_data.shape[0] - 2 - i*4,  # y = en bas de l'image, décalé vers le haut
            txt,
            ha='left',
            va='bottom',
            fontsize=8,
            color='yellow'
        )
    #plt.legend(fontsize=7)  
    plt.axis('off')
    plt.title('Phoneme: ' + str(phoneme) + ' - Frame: ' + str(name_file))
    plt.savefig(f'{folder_TV}/{name_file}.png', bbox_inches='tight')
    plt.close()
    
    
def _calculate_TV(arr1, arr2):
    TV_cdist = cdist(arr1, arr2, metric='euclidean')
    #TV_cdist = torch.cdist(arr1, arr2)

    # Trouver les valeurs minimales et leurs indices sur l'axe 0
    min_d0 = TV_cdist.min(axis=0)
    argmin_d0 = TV_cdist.argmin(axis=0)

    # Trouver le minimum de min_d0 et son indice
    min_d1 = min_d0.min()
    argmin_d1 = min_d0.argmin()

    # Trouver les indices finaux
    arr1_argmin = argmin_d0[argmin_d1]
    arr2_argmin = argmin_d1

    # Récupérer les points correspondants
    TV = min_d1
    TV_point_arr1 = arr1[arr1_argmin]
    TV_point_arr2 = arr2[arr2_argmin]
    return TV, TV_point_arr1, TV_point_arr2

def distance_point_line(Q, P0, v):
    # v doit être unitaire
    return np.linalg.norm((Q - P0) - np.dot(Q - P0, v) * v)


def _calculate_LP(lower_lip_arr, upper_lip_arr, uincisor_arr):
    point1 = uincisor_arr[6]
    min_lower_lip = np.min(lower_lip_arr[:, 0])
    min_upper_lip = np.min(upper_lip_arr[:, 0])
    min_x = min(min_lower_lip, min_upper_lip)
    point2 = np.array([[min_x, point1[1]]])  # On suppose que la coordonnée y est 0 pour le point de référence
    point1 = point1.reshape(1, -1)
    point2 = point2.reshape(1, -1)
    LP, LP_lip, LP_uincisor = _calculate_TV(point1, point2)
    return LP*1.62, LP_lip, LP_uincisor

def _calculate_LA(llip_arr, ulip_arr):
    LA, LA_llip, LA_ulip = _calculate_TV(llip_arr, ulip_arr)
    return LA*1.62, LA_llip, LA_ulip

def _calculate_LD(tongue_arr, uincisor_arr):
    tongue_tip_arr = tongue_arr[slice(*ART_SLICES["tongue-tip"])]
    teeth_arr = uincisor_arr[slice(*ART_SLICES["dental"])]
    TTCD, TTCD_tongue, TTDC_uincisor = _calculate_TV(tongue_tip_arr, teeth_arr)
    return TTCD*1.62, TTCD_tongue, TTDC_uincisor

def _calculate_TTCD(tongue_arr, uincisor_arr):
    tongue_tip_arr = tongue_arr[slice(*ART_SLICES["tongue-tip"])]
    teeth_arr = uincisor_arr[slice(*ART_SLICES["upper-incisor"])]
    TTCD, TTCD_tongue, TTDC_uincisor = _calculate_TV(tongue_tip_arr, teeth_arr)
    return TTCD*1.62, TTCD_tongue, TTDC_uincisor

def _calculate_TBCD(tongue_arr, uincisor_arr, soft_palate_velum_arr):
    tongue_body_arr = tongue_arr[slice(*ART_SLICES["tongue-body"])]
    hard_palate_arr = uincisor_arr[slice(*ART_SLICES["hard-palate"])]
    soft_palate_arr = soft_palate_velum_arr[slice(*ART_SLICES["soft-palate"])]
    palate_arr = np.concatenate([hard_palate_arr, soft_palate_arr], axis=0)
    TBCD, TBCD_tongue, TBCD_palate = _calculate_TV(tongue_body_arr, palate_arr)
    return TBCD*1.62, TBCD_tongue, TBCD_palate

def _calculate_TRCD(tongue_arr, pharynx_arr):
    tongue_root_arr = tongue_arr[slice(*ART_SLICES["tongue-root"])]
    TRCD, TRCD_tongue, TRCD_pharynx = _calculate_TV(tongue_root_arr, pharynx_arr)
    return TRCD*1.62, TRCD_tongue, TRCD_pharynx

def _calculate_TRCL(tongue_arr, soft_palate_arr):
    tongue_root_arr = tongue_arr[slice(*ART_SLICES["tongue-root"])]
    soft_palate_arr = soft_palate_arr[-1]
    soft_palate_arr = soft_palate_arr.reshape(1, -1)
    TRCL, TRCL_tongue, TRCL_soft_palate = _calculate_TV(tongue_root_arr, soft_palate_arr)
    return TRCL*1.62, TRCL_tongue, TRCL_soft_palate

def _calculate_PH(upper_incisor, vocal_arr, pharynx_arr):
    point1 = np.mean(vocal_arr, axis=0)
    y_target = upper_incisor[-1, 1]  # On prend le y de l'incisive supérieure la plus haute
    v = pharynx_arr[-1] - pharynx_arr[0]
    direction = v / np.linalg.norm(v) 
    
    dx, dy = direction
    if dy == 0:
        # Paroi horizontale (rare), on reste au même y → pas de solution unique
        raise ValueError("La paroi pharyngée est horizontale, impossible de projeter vers un y_target différent.")
    else:
        t = (y_target - point1[1]) / dy  # solve for t in: y_target = point1[1] + t * dy
        point2 = point1 + t * direction  # point2 = point on the parallel line with correct y
    point1 = point1.reshape(1, -1)
    point2 = point2.reshape(1, -1)
    
    PH, PH_vocal, PH_hard = _calculate_TV(point1, point2)

    return PH * 1.62, PH_vocal, PH_hard

def _calculate_VEL(soft_palate_arr, pca_c1, pca_scaler):
    soft_palate_arr_reshaped = soft_palate_arr.reshape(1,-1) 
    soft_palate_arr_reshaped = soft_palate_arr_reshaped[:,-50:]
    soft_palate_scaled = pca_scaler.transform(soft_palate_arr_reshaped)
    cp1 = pca_c1.components_[0]
    dot_products = (soft_palate_scaled @ cp1)
    sign = np.sign(dot_products)
    dot_products = dot_products[0]
    # soft_palate_projected = pca_c1.transform(soft_palate_scaled)
 
    # #centered = soft_palate_arr_reshaped - pca_mean
    # scalar_proj = np.dot(soft_palate_projected, pca_c1)  # => un seul nombre
    sign = "positif" if dot_products >= 0 else "négatif"

    if sign == "positif":
        return dot_products * 1.62, np.array([1, 1]), soft_palate_arr[-1]
    else :
        return dot_products * 1.62, np.array([-1, -1]), soft_palate_arr[-1]

# def plot_VEL(soft_palate_arr, pca_c1, pca_scaler, frame_name, phoneme, folder_run, prefix):
#     soft_palate_arr_reshaped = soft_palate_arr.reshape(-1) 
#     # centered = soft_palate_arr_reshaped - pca_mean
#     # X_proj_PC1 = (centered @ pca_c1)* pca_c1 + pca_mean
#     soft_palate_scaled = pca_scaler.transform(soft_palate_arr_reshaped)
#     soft_palate_projected = pca_c1.transform(soft_palate_scaled)
#     X_proj_PC1 = soft_palate_projected.reshape(50,2)
#     scalar_proj = np.dot(soft_palate_projected, pca_c1)  # => un seul nombre
#     sign = "positif" if scalar_proj >= 0 else "négatif"
    
    
#     name_file = f"{int(frame_name[0])}_S{int(frame_name[1])}_{int(frame_name[2]):04d}"
#     image_path = f"/srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/ArtSpeech_Database_2/{int(frame_name[0])}/S{int(frame_name[1])}/NPY_MR_registered/{int(frame_name[2]):04d}.npy"
#     image_data = np.load(image_path)
#     folder_velum_image = os.path.join(folder_run, f'velum_image_{prefix}')
#     os.makedirs(folder_velum_image, exist_ok=True)  # Crée le dossier s'il n'existe pas
    
    

#     folder_velum = os.path.join(folder_run, f'velum_{prefix}')
#     os.makedirs(folder_velum, exist_ok=True)  # Crée le dossier s'il n'existe pas
    
#     plt.figure(figsize=(8, 6))
#     plt.imshow(image_data, cmap='gray')
#     plt.plot(soft_palate_arr[:, 0], soft_palate_arr[:, 1],c='skyblue',  label='Original Data')
#     plt.plot(X_proj_PC1[:, 0], X_proj_PC1[:, 1], c='darkblue', label='Projection on PC1')
#     plt.legend()
#     plt.axis('off')
    
#     info_text = (
#         f"Produit scalaire : {scalar_proj:.2f} ({sign})"
#         )

#     # Texte informatif
#     plt.text(0.03, 0.06,
#         info_text,
#         transform=plt.gca().transAxes,
#         fontsize=10,
#         verticalalignment='top',
#         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    
#     plt.title('Phoneme: ' + str(phoneme) + ' - Frame: ' + str(name_file))
#     plt.savefig(f'{folder_velum_image}/{name_file}.png', bbox_inches='tight')
#     plt.close()
    
    
#     plt.figure(figsize=(8, 6))
#     plt.scatter(soft_palate_arr[:, 0], soft_palate_arr[:, 1],c='skyblue',  label='Original Data')
#     plt.scatter(X_proj_PC1[:, 0], X_proj_PC1[:, 1], c='darkblue', label='Projection on PC1')
#     for j in range(50):
#             plt.plot(
#                 [soft_palate_arr[j, 0], X_proj_PC1[j, 0]],
#                 [soft_palate_arr[j, 1], X_proj_PC1[j, 1]],
#                 color='gray', linewidth=0.8
#             )
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.xlim(45,75)
#     plt.ylim(52,75)
#     plt.gca().invert_yaxis()
#     plt.legend()

    
#     info_text = (
#         f"Produit scalaire : {scalar_proj:.2f} ({sign})"
#         )

#     # Texte informatif
#     plt.text(0.03, 0.06,
#         info_text,
#         transform=plt.gca().transAxes,
#         fontsize=10,
#         verticalalignment='top',
#         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    
#     plt.title('Phoneme: ' + str(phoneme) + ' - Frame: ' + str(name_file))
#     plt.savefig(f'{folder_velum}/{name_file}.png', bbox_inches='tight')
#     plt.close()


def plot_VEL(soft_palate_arr, pharynx_arr, pca_c1, pca_scaler, frame_name, phoneme, folder_run, prefix):
    soft_palate_arr_reshaped = soft_palate_arr.reshape(1,-1) # Prendre les 25 derniers points
    soft_palate_arr_reshaped = soft_palate_arr_reshaped[:,-50:]  # Prendre les 50 derniers points
    soft_palate_scaled = pca_scaler.transform(soft_palate_arr_reshaped)
    
    cp1 = pca_c1.components_[0]  # shape: (n_features,)
    # cp2 = pca_c1.components_[1]  # shape: (n_features,)

    # # Produit scalaire avec chaque composante
    # dot1 = float(soft_palate_scaled @ cp1.T)  # scalaire
    # dot2 = float(soft_palate_scaled @ cp2.T)  # scalaire

    # dot_products = dot1+ dot2
    dot_products = float(soft_palate_scaled @ cp1)
    signs = np.sign(dot_products)
    #X_test_proj_scaled = np.outer(dot_products, cp1)
    # soft_palate_pca_proj = pca_c1.transform(soft_palate_scaled)
    # X_test_proj_scaled = pca_c1.inverse_transform(soft_palate_pca_proj)
    # X_test_proj_orig = X_test_proj_scaled * pca_scaler.scale_ + pca_scaler.mean_
    soft_palate_scaled = pca_scaler.transform(soft_palate_arr_reshaped)  # Standardisation

    # Projection complète dans l'espace PCA
    soft_palate_pca_proj = pca_c1.transform(soft_palate_scaled)  # shape: (1, n_components)

    # Garder uniquement les 2 premières composantes, mettre les autres à 0
    soft_palate_pca_proj_reduced = np.zeros_like(soft_palate_pca_proj)
    soft_palate_pca_proj_reduced[:, :2] = soft_palate_pca_proj[:, :2]

    # Reconstruction dans l'espace standardisé
    X_test_proj_scaled = pca_c1.inverse_transform(soft_palate_pca_proj_reduced)  # shape: (1, n_features)

    # Revenir à l'espace original non standardisé
    X_test_proj_orig = X_test_proj_scaled * pca_scaler.scale_ + pca_scaler.mean_

    name_file = f"{int(frame_name[0])}_S{int(frame_name[1])}_{int(frame_name[2]):04d}"
    image_path = f"/srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/ArtSpeech_Database_2/{int(frame_name[0])}/S{int(frame_name[1])}/NPY_MR_registered/{int(frame_name[2]):04d}.npy"
    image_data = np.load(image_path)
    #folder_run = '/srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/pca_test'
    folder_velum_image = os.path.join(folder_run, f'velum_image_{prefix}')
    os.makedirs(folder_velum_image, exist_ok=True)  # Crée le dossier s'il n'existe pas
    
    

    folder_velum = os.path.join(folder_run, f'velum_{prefix}')
    os.makedirs(folder_velum, exist_ok=True)  # Crée le dossier s'il n'existe pas
    X_test_proj_orig = X_test_proj_orig.reshape(25,2)
    plt.figure()
    plt.imshow(image_data, cmap='gray')
    plt.plot(pharynx_arr[:, 0], pharynx_arr[:, 1],c='goldenrod')#, label='Pharyngeal wall')
    plt.plot(soft_palate_arr[-25:, 0], soft_palate_arr[-25:, 1],c='skyblue')#, label='Original Data')
    plt.plot(X_test_proj_orig[:, 0], X_test_proj_orig[:, 1], c='darkblue')#, label='Projection on PC1')
    # for j in range(50):
    #         plt.plot(
    #             [soft_palate_arr[j, 0], X_proj_PC1[j, 0]],
    #             [soft_palate_arr[j, 1], X_proj_PC1[j, 1]],
    #             color='gray', linewidth=0.3
    #         )
    #plt.legend()

    
    
    info_text = (
        f"Dot product: {dot_products:.2f}\n"
        f"Phoneme: /{phoneme[0]}/"
        )

    # Texte informatif
    plt.text(
        2,                      # x = 2 pixels depuis la gauche
        image_data.shape[0] -6 ,  # y = en bas de l'image, décalé vers le haut
        info_text,
        ha='left',
        va='bottom',
        fontsize=10,
        color='yellow'
    )

    plt.grid(False)
    plt.axis('off')
    # plt.title('Phoneme: ' + str(phoneme) + ' - Frame: ' + str(name_file))
    plt.savefig(f'{folder_velum_image}/{name_file}.png', bbox_inches='tight')
    plt.close()
    
    
    plt.figure()
    plt.scatter(pharynx_arr[:, 0], pharynx_arr[:, 1],c='goldenrod')#, label='Pharyngeal wall')
    plt.scatter(soft_palate_arr[:, 0], soft_palate_arr[:, 1],c='skyblue')#, label='Original Data')
    plt.scatter(X_test_proj_orig[:, 0], X_test_proj_orig[:, 1], c='darkblue')#, label='Projection on PC1')
    for j in range(25):
            plt.plot(
                [soft_palate_arr[j, 0], X_test_proj_orig[j, 0]],
                [soft_palate_arr[j, 1], X_test_proj_orig[j, 1]],
                color='gray', linewidth=0.8
            )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(45,75)
    plt.ylim(52,75)
    plt.gca().invert_yaxis()
    plt.legend()

    
    # Texte informatif
    plt.text(
        2,                      # x = 2 pixels depuis la gauche
        image_data.shape[0] -6 ,  # y = en bas de l'image, décalé vers le haut
        info_text,
        ha='left',
        va='bottom',
        fontsize=10,
        color='yellow'
    )

    plt.grid(False)
    plt.axis('off')
    # plt.title('Phoneme: ' + str(phoneme) + ' - Frame: ' + str(name_file))
    plt.savefig(f'{folder_velum}/{name_file}.png', bbox_inches='tight')
    plt.close()
    
    
         
def calculate_vocal_tract_variables(inputs_dict, frame_name, phoneme, folder_run, pca_c1, pca_scaler, prefix=""):
    """
    Args:
        inputs_dict (dict): Dictionary containing the articulator name as key and the articulator points as value.
    Return:
        TVs (dict): Dictionary containing the TV name as key and the value and location as value.

    Measured vocal tract variables are:
    LA - Lip aperture
    LP - Lip protusion
    TTCD - Tongue tip constrict degree
    TTCL - Tongue tip constrict location
    TBCD - Tongue body constrict degree
    TBCL - Tongue body constrict location
    VEL - Velum
    GLO - Glottis
    """
    
    LP, LP_lip, LP_uincisor = _calculate_LP(inputs_dict[LOWER_LIP], inputs_dict[UPPER_LIP], inputs_dict[UPPER_INCISOR])
    LA, LA_llip, LA_ulip = _calculate_LA(inputs_dict[LOWER_LIP], inputs_dict[UPPER_LIP])
    LD, LD_tongue, LD_uincisor = _calculate_LD(inputs_dict[TONGUE], inputs_dict[UPPER_INCISOR])
    
    TTCD, TTCD_tongue, TTDC_uincisor = _calculate_TTCD(inputs_dict[TONGUE], inputs_dict[UPPER_INCISOR])
    TBCD, TBCD_tongue, TBCD_palate = _calculate_TBCD(inputs_dict[TONGUE], inputs_dict[UPPER_INCISOR], inputs_dict[SOFT_PALATE_MIDLINE])
    TRCD, TRCD_tongue, TRCD_pharynx = _calculate_TRCD(inputs_dict[TONGUE], inputs_dict[PHARYNX])
    TRCL, TRCL_tongue, TRCL_soft_palate = _calculate_TRCL(inputs_dict[TONGUE], inputs_dict[SOFT_PALATE_MIDLINE])
    
    PH, PH_vocal, PH_velum = _calculate_PH(inputs_dict[UPPER_INCISOR], inputs_dict[VOCAL_FOLDS], inputs_dict[PHARYNX])
    VEL, VEL_velum, VEL_pharynx = _calculate_VEL(inputs_dict[SOFT_PALATE_MIDLINE], pca_c1, pca_scaler)
    
    

    # PoC stands for Place of Constriction. For each TV, there are two PoCs,
    # from which the TV is measured.
    # TODO: Implement the functions to calculate LP, TTCL, TBCL, GLO
    TVs = {
        "LP": {
            "value": LP,
            "poc_1": LP_lip,
            "poc_2": LP_uincisor
        },
        "LA": {
            "value": LA,
            "poc_1": LA_llip,
            "poc_2": LA_ulip
        },
        "LD": {
            "value": LD,
            "poc_1": LD_tongue,
            "poc_2": LD_uincisor
        },
        "TTCD": {
            "value": TTCD,
            "poc_1": TTCD_tongue,
            "poc_2": TTDC_uincisor
        },
        "TBCD": {
            "value": TBCD,
            "poc_1": TBCD_tongue,
            "poc_2": TBCD_palate
        },
        "TRCD": {
            "value": TRCD,
            "poc_1": TRCD_tongue,
            "poc_2": TRCD_pharynx
        },
        "TRCL":{
            "value": TRCL,
            "poc_1": TRCL_tongue,
            "poc_2": TRCL_soft_palate
        },
        "PH": {
            "value": PH,
            "poc_1": PH_vocal,
            "poc_2": PH_velum
        },
        "VEL": {
            "value": VEL,
            "poc_1": VEL_velum,
            "poc_2": VEL_pharynx
        },
        
    }
    
    # if int(frame_name[0]) == 1775 and int(frame_name[1]) == 20 :
    #     plot_vt(inputs_dict, frame_name, phoneme, TVs, folder_run, prefix)
    #     plot_VEL(inputs_dict[SOFT_PALATE_MIDLINE], inputs_dict[PHARYNX],pca_c1, pca_scaler, frame_name, phoneme, folder_run, prefix)
    # TVs = []
    return TVs
