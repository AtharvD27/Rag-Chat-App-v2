import argparse
import warnings
from utils import load_config
from run_chat import (
    load_documents, update_vectorstore,
    setup_llm, handle_session, start_session
)

def main():
    parser = argparse.ArgumentParser(description="RAG Chat Interface")
    parser.add_argument("--model_path", type=str, help="Override model path")
    parser.add_argument("--temperature", type=float, help="Override LLM temperature")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--skip_update", action="store_true", help="Skip vectorstore update")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode and show warnings.")


    args = parser.parse_args()
    
    if not args.debug:
        warnings.filterwarnings("ignore")

    config = load_config(args.config)
    overrides = {
        "model_path": args.model_path,
        "temperature": args.temperature
    }

    # Chat session
    print("🤖 Starting RAG chat agent...")
    chunks = load_documents(config)
    retriever = update_vectorstore(config, chunks, skip_update=args.skip_update)
    llm = setup_llm(config, overrides)

    config["retriever"] = retriever
    config["llm_instance"] = llm

    snap, memory, = handle_session(config)
    agent = start_session(config, memory)

    print("\n🔍 Ask questions. Type 'exit' to quit, or use ::new / ::resume\n")
    has_activity = False
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            if has_activity:
                snap.save_snapshot()
                print("✅ Session saved before exit.\n")
            else:
                print("🗑️ No activity detected. Session discarded.\n")
            break
        
        elif user_input.lower() == "::new":
            if has_activity:
                snap.save_snapshot()
                print("💾 Previous session saved.\n")
            snap, memory= handle_session(config, override="new")
            agent = start_session(config, memory)
            has_activity = False
            continue
        
        elif user_input.lower() == "::resume":
            if has_activity:
                snap.save_snapshot()
                print("💾 Previous session saved.\n")
            snap, memory= handle_session(config, override="resume")
            agent = start_session(config, memory)
            has_activity = False
            continue
        
        answer, sources = agent.ask(user_input)
        print(f"\n\n💬 Answer:\n{answer}\n\n📚 Sources:")
        for s in sources:
            print(f" - {s['file']} (Page {s['page']}, Chunk {s['chunk']}):\n {s['text'][:150]}...\n")
        snap.record_turn(user_input, answer, sources)
        has_activity = True

if __name__ == "__main__":
    main()
