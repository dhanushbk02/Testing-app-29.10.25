from google.oauth2 import service_account
from googleapiclient.discovery import build

# --- Load Google Service Account credentials ---
SERVICE_ACCOUNT_FILE = "service_account.json"  # Path to your JSON key
SCOPES = ["https://www.googleapis.com/auth/drive"]

creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)

# --- Connect to Google Drive API ---
service = build("drive", "v3", credentials=creds)

# --- Google Drive folder ID (your actual folder) ---
FOLDER_ID = "1fBbVWcyCrvwF5pWZhXsjRYfVEKNYtV0q"

# --- List files inside your folder ---
results = (
    service.files()
    .list(
        q=f"'{FOLDER_ID}' in parents",
        fields="files(id, name, mimeType, modifiedTime)"
    )
    .execute()
)
items = results.get("files", [])

if not items:
    print("No files found in your folder.")
else:
    print("Files in folder:")
    for item in items:
        print(f"{item['name']} ({item['id']})")
