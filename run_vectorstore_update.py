import argparse
import warnings
from tqdm import tqdm
from utils import load_config
from vectorstore_manager import VectorstoreManager
from document_loader import SmartDocumentLoader

# CLI setup
parser = argparse.ArgumentParser(description="Manage vectorstore lifecycle.")
parser.add_argument("--update", action="store_true", help="Update vectorstore with only new documents.")
parser.add_argument("--delete", action="store_true", help="Delete the existing vectorstore.")
parser.add_argument("--reset", action="store_true", help="Delete and rebuild the vectorstore.")
parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file.")
parser.add_argument("--debug", action="store_true", help="Enable debug mode and show warnings.")

args = parser.parse_args()

if not args.debug:
        warnings.filterwarnings("ignore")

if not (args.update or args.delete or args.reset):
    parser.print_help()
    exit(0)

# Load config
config = load_config(args.config)
vs_manager = VectorstoreManager(config)

# DELETE operation
if args.delete:
    vs_manager.delete_vectorstore()
    exit(0)

# RESET operation
if args.reset:
    print("🔄 Resetting vectorstore...")
    vs_manager.delete_vectorstore()

# Use Smart Loader
loader = SmartDocumentLoader(config=config)
documents = loader.load()
#print(f"📄 Loaded {len(documents)} documents from multiple sources.")

# Chunking
#print("🔪 Splitting into chunks...")
chunks = loader.split_documents(documents)

# Add to vectorstore
vs_manager.load_vectorstore()
if vs_manager.needs_update(chunks):
    vs_manager.add_documents(chunks)
else:
    print("✅ Your knowledge base is already up to date.")

