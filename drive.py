from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import argparse


def get_params():
    parser=argparse.ArgumentParser(description="Pass the configurations here")
    parser.add_argument("--datapath",type=str,help = "if upload - provide the file path| \n if download provide the filename to save")
    parser.add_argument("--action",type=str,help = "upload/download")
    return vars(parser.parse_args())


def Oauth():
    gauth = GoogleAuth()
    gauth.CommandLineAuth()

    drive = GoogleDrive(gauth)

    return drive

def Upload(args,drive):
    file = drive.CreateFile()
    file.SetContentFile(args['datapath'])
    file.Upload()
    print('Created file %s with mimeType %s' % (file['title'],file['mimeType']))
    
def Download(args,drive):
    id = str(input("Enter the file id ::: "))
    # exec(gdown --id id)
    file = drive.CreateFile({'id': id})
    print(file['title'],file['mimeType'])
    mimetypes = {
        # Drive Document files as PDF
        'application/vnd.google-apps.document': 'application/pdf',

        # Drive Sheets files as MS Excel files.
        'application/vnd.google-apps.spreadsheet': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'}
    download_mimetype = None
    
    try:
        download_mimetype = mimetypes[file['mimeType']]
    except:
        pass

    print('Downloading file %s from Google Drive' % file['title'])
    file.GetContentFile(args['datapath'], mimetype=download_mimetype)


if __name__ == "__main__":

    args = get_params()
    drive = Oauth()

    if args["action"].lower() == "upload":

        Upload(args,drive)

    elif args["action"].lower() == "download":

        Download(args,drive)
    else:
        print("your action is not identified \n make sure you typed upload/download")




