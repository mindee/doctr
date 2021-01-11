import os
import fitz 
import magic
import numpy as np
import cv2

ALLOWED_PDF = ["application/pdf"]
DEFAULT_RES_MIN = 0.8e6
DEFAULT_RES_MAX = 3e6


def documents_to_strings(documents_imgs):
    """
    :param documents_imgs: list of documents as np imgs, a document is either a np array or a list of np arrays (multiple pages)
    :returns: - list of int : heights
              - list of int : widths
              - list of binary string images
    All 3 lists have the same nested structure for documents : list of list for pages of the same doc
    """

    heights = []
    widths = []
    raw_images = []

    for document in documents_imgs:
        if not isinstance(document, list):
            document = [document]

        heights.append([page.shape[0] for page in document])
        widths.append([page.shape[1] for page in document])
        raw_images.append([page.flatten().tostring() for page in document])
    
    return heights, widths, raw_images



def prepare_pdf_documents(
    filepaths=None, pdf_resolution=None, with_sizes=False
):
    """
    Always return tuple of:
        - list of documents, each doc is a numpy image pages list (valid RGB image with 3 channels)
        - list of document names, each page inside a doc has a different name
    optional : list of sizes
    :param filepaths: list of pdf filepaths to prepare, or a filepath (str)
    :param pdf_resolution: output resolution of images
    :param with_sizes: to return the list of sizes
    """

    if filepaths is None:
        raise Exception

    documents_imgs = []
    documents_names = []
    documents_sizes = []

    # make document dimension if not existing:
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    for f_document in filepaths:

        size = os.path.getsize(f_document)
        pages_imgs, pages_names = prepare_pdf_from_filepath(
            f_document, pdf_resolution=pdf_resolution
        )

        documents_imgs.append(pages_imgs)
        documents_names.append(pages_names)
        documents_sizes.append(size)

    if with_sizes:
        return documents_imgs, documents_names, documents_sizes

    return documents_imgs, documents_names

 

def prepare_pdf_from_filepath(
    filepath, pdf_resolution=None
):
    """
    Read a pdf from a filepath with fitz
    :param filepath: filepath of the .pdf file
    :param pdf_resolution: output resolution
    """

    if not os.path.isfile(filepath):
        raise FileNotFoundError
    
    filename = os.path.splitext(os.path.split(filepath)[-1])[0]
    mimetype = magic.from_file(filepath, True)

    if mimetype in ALLOWED_PDF:
        try:
            pdf = fitz.open(filepath)

        except:
            return None

        imgs, names = convert_pdf_pages_to_imgs(pdf, filename, resolution=pdf_resolution)
        return imgs, names
    
    else:
        raise NameError('not a pdf')
 

def convert_pdf_pages_to_imgs(
    pdf, filename, pages=None, resolution=None, img_type="np"
):
    """
    Convert pdf pages to numpy arrays.
    :param pdf: pdf doc opened with fitz
    :param filename: pdf name to rename pages 
    :param img_type: The format of the output pages, can be "np" or "png"
    :param pages: Int or list of int to specify which pages to take. If None, takes all pages.
    :param resolution: Output resolution in pixels. If None, use the default page size (DPI@96).
    Can be used as a tuple to force a minimum/maximum resolution dynamically.
    :param with_names: Output list of names in return statement.
    :return: List of numpy arrays of dtype uint8.
    """

    imgs = []
    names = []

    # Decode pages parameter
    if isinstance(pages, int):
        pages = [pages]
    pages = pages or [x + 1 for x in range(len(pdf))]

    # Iterate over pages
    for i, page in enumerate(pdf):
        if i + 1 not in pages:
            continue
        
        #set resolution
        out_res = set_resolution(resolution, res_min=DEFAULT_RES_MIN, res_max=DEFAULT_RES_MAX)

        # Make numpy array
        pixmap = page_to_pixmap(page, out_res)

        if img_type == "np":
            imgs.append(pixmap_to_numpy(pixmap))
        elif img_type == "png":
            imgs.append(pixmap.getImageData(output="png"))
        else:
            logger.warning(f"could not convert to {img_type}, returning png")
            imgs.append(pixmap.getImageData(output="png"))

        names.append(f"{filename}-p{str(i + 1).zfill(3)}")

    return imgs, names


def set_resolution(resolution, res_min, res_max):
    #resolution : None, int or tuple
    out_res = resolution

    if out_res is None:
        return out_res

    if isinstance(resolution, tuple):
        assert len(resolution) == 2
        out_res = resolution[0] * resolution[1]
 
    #bound resolution     
    if out_res < res_min:
        return res_min
    elif out_res > res_max:
        return res_max
    else:
        return out_res


def page_to_pixmap(page, resolution=None):
    out_res = resolution
    scale = 1  # internal DPI is always 96
    if out_res:
        box = page.MediaBox
        in_res = int(box[2]) * int(box[3])
        scale = min(20, out_res / in_res) #to prevent error if in_res is very low
    return page.getPixmap(matrix=fitz.Matrix(scale, scale))

def pixmap_to_numpy(pixmap, channel="RGB"):
    stream = pixmap.getImageData()
    stream = np.frombuffer(stream, dtype=np.uint8)
    img = cv2.imdecode(stream, cv2.IMREAD_UNCHANGED)  
    if channel == "RGB":
        return img[:, :, ::-1]
    elif channel == "BGR":
        return img
    else:
        raise Exception("Invalid channel parameter! Must be RGB or BGR")


"""
filepaths = ["/home/datascientist-4/seg_dataset/sample/00ab1a5d-a1b0-4f68-90aa-a03955b0095b.pdf",
"/home/datascientist-4/seg_dataset/sample/00b443c8-dc34-442d-9c48-fab08a9b74cf.pdf",
"/home/datascientist-4/seg_dataset/sample/00b3352d-4a0a-4fa6-8b98-6e15b8cdc4cb.pdf", 
"/home/datascientist-4/ex.pdf"]

images, names, sizes = prepare_pdf_documents(
    filepaths=filepaths, pdf_resolution=None, with_sizes=True)

print(images)
print(names)
print(sizes)

heights, widths, stri =  documents_to_strings(images)

print(heights)
print(widths)
"""