import torchxrayvision as xrv

NIH_IMAGES = "/run/user/1000/gvfs/smb-share:server=lxestudios.hospitalitaliano.net,share=pacs/T-Rx/NIH/"
CHEXPERT_IMAGES = "/run/user/1000/gvfs/smb-share:server=lxestudios.hospitalitaliano.net,share=pacs/T-Rx/CheXpert-v1.0-small"
CHEXPERT_CSV = "/run/user/1000/gvfs/smb-share:server=lxestudios.hospitalitaliano.net,share=pacs/T-Rx/CheXpert-v1.0-small/train_renamed_ubuntu.csv"
NIH_GOOGLE_IMAGES = "/home/mila/c/cohenjos/data/images-224-NIH"
PADCHEST_IMAGES = "/home/mila/c/cohenjos/data/images-224-PC"
MIMIC_IMAGES = "/lustre04/scratch/cohenjos/MIMIC/images-224/files"
MIMIC_CSV = "/lustre03/project/6008064/jpcohen/MIMICCXR-2.0/mimic-cxr-2.0.0-chexpert.csv.gz"
MIMIC_METADATA = "/lustre03/project/6008064/jpcohen/MIMICCXR-2.0/mimic-cxr-2.0.0-metadata.csv.gz"
OPENI_IMAGES = "/lustre03/project/6008064/jpcohen/OpenI/images/"
RSNA_IMAGES = "/lustre03/project/6008064/jpcohen/kaggle-pneumonia/stage_2_train_images_jpg"


def parseDatasets(datasets,transforms,data_aug):
    datas = []
    datas_names = []
    if "nih" in datasets:
        dataset = xrv.datasets.NIH_Dataset(
            imgpath=NIH_IMAGES,
            transform=transforms, data_aug=data_aug,
            views=['PA',
                   'AP'
                   ],
            unique_patients=True,
                )
        datas.append(dataset)
        datas_names.append("nih")
    if "pc" in datasets:
        dataset = xrv.datasets.PC_Dataset(
            imgpath=PADCHEST_IMAGES,
            transform=transforms, data_aug=data_aug)
        datas.append(dataset)
        datas_names.append("pc")
    if "chex" in datasets:
        dataset = xrv.datasets.CheX_Dataset(
            imgpath=CHEXPERT_IMAGES,
            csvpath=CHEXPERT_CSV,
            transform=transforms, data_aug=data_aug,
            views=['PA', 'AP'],
            unique_patients=True,
        )
        datas.append(dataset)
        datas_names.append("chex")
    if "google" in datasets:
        dataset = xrv.datasets.NIH_Google_Dataset(
            imgpath=NIH_GOOGLE_IMAGES,
            transform=transforms, data_aug=data_aug)
        datas.append(dataset)
        datas_names.append("google")
    if "mimic_ch" in datasets:
        dataset = xrv.datasets.MIMIC_Dataset(
            imgpath=MIMIC_IMAGES,
            csvpath=MIMIC_CSV,
            metacsvpath=MIMIC_METADATA,
            transform=transforms, data_aug=data_aug)
        datas.append(dataset)
        datas_names.append("mimic_ch")
    if "openi" in datasets:
        dataset = xrv.datasets.Openi_Dataset(
            imgpath=OPENI_IMAGES,
            transform=transforms, data_aug=data_aug)
        datas.append(dataset)
        datas_names.append("openi")
    if "rsna" in datasets:
        dataset = xrv.datasets.RSNA_Pneumonia_Dataset(
            imgpath=RSNA_IMAGES,
            transform=transforms, data_aug=data_aug)
        datas.append(dataset)
        datas_names.append("rsna")
    return datas, datas_names
