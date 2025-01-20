import sys
import readline
from pathlib import Path
from typing import List, Optional
from colorama import init, Fore, Style
from tqdm import tqdm

from src.core import qa_service, pdf_parser, indexing_service, embedding_service
from src.config.settings import IS_DEV

# Initialize colorama for cross-platform colored output
init()

class CLI:
    def __init__(self):
        self.pdf_parser = pdf_parser.PDFParser()
        self.embedding_service = embedding_service.EmbeddingService()
        self.indexing_service = indexing_service.IndexingService()
        self.qa_service = qa_service.QAService()
        self.command_history = []
        
        # Configure readline with all standard bindings
        readline.parse_and_bind('"\e[A": previous-history')    # Up arrow
        readline.parse_and_bind('"\e[B": next-history')       # Down arrow
        readline.parse_and_bind('"\e[C": forward-char')       # Right arrow
        readline.parse_and_bind('"\e[D": backward-char')      # Left arrow
        readline.parse_and_bind('"\C-w": unix-word-rubout')   # Ctrl+W to delete word
        readline.parse_and_bind('"\C-u": unix-line-discard')  # Ctrl+U to clear line
        readline.parse_and_bind('"\C-a": beginning-of-line')  # Ctrl+A to start of line
        readline.parse_and_bind('"\C-e": end-of-line')        # Ctrl+E to end of line
        readline.parse_and_bind('set enable-keypad on')       # Enable keypad keys
        readline.parse_and_bind('set input-meta on')          # Enable meta key
        readline.parse_and_bind('set convert-meta off')       # Don't convert meta characters
        readline.parse_and_bind('set output-meta on')         # Output meta characters as-is
        readline.set_history_length(1000)                     # Set history size
        
    def print_welcome(self):
        """Print welcome message and available commands."""
        print(f"{Fore.CYAN}Welcome to Scientific Paper Parser!{Style.RESET_ALL}")
        self.print_help()

    def print_help(self):
        """Print available commands."""
        print("\nAvailable commands:")
        print(f"{Fore.GREEN}ingest <folder>{Style.RESET_ALL} - Parse and index PDFs from a folder")
        print(f"{Fore.GREEN}ask <question>{Style.RESET_ALL} - Ask a question about the indexed papers")
        print(f"{Fore.GREEN}status{Style.RESET_ALL} - Show OpenSearch index statistics")
        print(f"{Fore.GREEN}help{Style.RESET_ALL} - Show this help message")
        print(f"{Fore.GREEN}exit{Style.RESET_ALL} - Exit the application")
        if IS_DEV:
            print(f"{Fore.GREEN}reload{Style.RESET_ALL} - Hot reload python code for local development")
            print(f"{Fore.GREEN}invalidate <file/folder>{Style.RESET_ALL} - Remove PDFs from the index")

    def ingest_folder(self, folder_path: str):
        """Ingest all PDFs from a folder."""
        try:
            folder = Path(folder_path)
            if not folder.exists():
                print(f"{Fore.RED}Error: Folder does not exist{Style.RESET_ALL}")
                return

            pdf_files: List[Path] = list(folder.glob("*.pdf"))
            if not pdf_files:
                print(f"{Fore.YELLOW}No PDF files found in the folder{Style.RESET_ALL}")
                return

            # Calculate checksums first
            file_checksums: dict[Path, str] = {
                pdf_file: self.pdf_parser.compute_checksum(pdf_file) 
                for pdf_file in pdf_files
            }

            # Check which checksums exist in one query
            existing_checksums = self.indexing_service.check_existing_checksums(list(file_checksums.values()))

            # Separate new and skipped files
            new_pdfs: List[Path] = [
                pdf_file for pdf_file, checksum in file_checksums.items() 
                if checksum not in existing_checksums
            ]

            # Get the checksums of the existing PDFs, we can remove this later since its not really used
            skipped_pdfs: List[str] = [
                pdf_file.name for pdf_file, checksum in file_checksums.items() 
                if checksum in existing_checksums
            ]

            print(f"\nFound {len(pdf_files)} PDF files:")
                        
            if skipped_pdfs:
                print("\nSkipped PDFs (already indexed):")
                for pdf in skipped_pdfs:
                    print(f"- {pdf}")

            if not new_pdfs:
                print(f"\n{Fore.YELLOW}No new PDFs to process.{Style.RESET_ALL}")
                return

            print(f"\nProcessing {len(new_pdfs)} new PDF files...")
            
            success_count = 0
            error_count = 0
            errors = []
            
            for pdf_file in tqdm(new_pdfs, desc="Processing PDFs"):
                try:
                    # Use the pre-computed checksum from file_checksums
                    chunks = self.pdf_parser.parse_pdf(
                        pdf_file, 
                        file_checksums[pdf_file]
                    )
                    
                    # Generate embeddings for chunks
                    chunks = self.embedding_service.embed_chunks(chunks)
                    
                    # Index chunks in OpenSearch
                    self.indexing_service.index_chunks(chunks)
                    
                    success_count += 1
                    
                except Exception as e:
                    error_count += 1
                    errors.append(f"{pdf_file.name}: {str(e)}")
                    print(f"{Fore.RED}Error processing {pdf_file.name}: {str(e)}{Style.RESET_ALL}")
                    continue

            # Final status report
            if success_count == len(new_pdfs):
                print(f"{Fore.GREEN}Successfully processed all {success_count} new PDF files!{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.RED}Ingestion completed with errors:{Style.RESET_ALL}")
                print(f"Successful: {success_count}")
                print(f"Failed: {error_count}")
                print("\nErrors encountered:")
                for error in errors:
                    print(f"{Fore.RED}- {error}{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}Fatal error during ingestion: {str(e)}{Style.RESET_ALL}")

    def ask_question(self, question: str):
        """Ask a question about the indexed papers."""
        try:
            answer = self.qa_service.answer_question(question)
            print(f"\n{Fore.CYAN}Answer:{Style.RESET_ALL}")
            print(answer)
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")

    def show_index_status(self):
        """Show OpenSearch index statistics."""
        try:
            stats = self.indexing_service.get_index_stats()
            print(f"\n{Fore.CYAN}OpenSearch Index Status:{Style.RESET_ALL}")
            print(f"Total Documents: {stats['doc_count']}")
            print(f"Index Size: {stats['store_size']}")
            
            # Show sample documents if available
            sample_docs = self.indexing_service.get_sample_documents(5)
            if sample_docs:
                print(f"\n{Fore.CYAN}Sample Documents:{Style.RESET_ALL}")
                for doc in sample_docs:
                    print(f"\nID: {doc['_id']}")
                    print(f"Source: {doc['_source'].get('source', 'N/A')}")
                    print(f"Text Preview: {doc['_source'].get('text', 'N/A')[:200]}...")
                
        except Exception as e:
            print(f"{Fore.RED}Error getting index status: {str(e)}{Style.RESET_ALL}")

    def invalidate_documents(self, path: str):
        """Invalidate (delete) documents from the index."""
        try:
            target = Path(path)
            documents_to_invalidate = []

            if target.is_file():
                if target.suffix.lower() == '.pdf':
                    documents_to_invalidate.append(target.name)
                else:
                    print(f"{Fore.RED}Error: Not a PDF file: {target}{Style.RESET_ALL}")
                    return
            elif target.is_dir():
                # Get all PDF files in directory
                documents_to_invalidate = [
                    pdf.name for pdf in target.glob("*.pdf")
                ]
            else:
                print(f"{Fore.RED}Error: Path does not exist: {target}{Style.RESET_ALL}")
                return

            if not documents_to_invalidate:
                print(f"{Fore.YELLOW}No PDF files found to invalidate.{Style.RESET_ALL}")
                return

            # Confirm with user
            print(f"\nFound {len(documents_to_invalidate)} documents to invalidate:")
            for doc in documents_to_invalidate:
                print(f"- {doc}")
            
            confirm = input(f"\n{Fore.YELLOW}Are you sure you want to invalidate these documents? (y/N): {Style.RESET_ALL}").lower()
            if confirm != 'y':
                print("Operation cancelled.")
                return

            # Delete documents
            print("\nInvalidating documents...")
            result = self.indexing_service.delete_by_document_ids(documents_to_invalidate)
            
            # Report results
            print(f"\n{Fore.GREEN}Invalidation complete:{Style.RESET_ALL}")
            print(f"Documents processed: {len(documents_to_invalidate)}")
            print(f"Chunks deleted: {result['total_deleted']}")
            
            if result['total_failed']:
                print(f"{Fore.RED}Failed deletions: {result['total_failed']}{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}Error during invalidation: {str(e)}{Style.RESET_ALL}")

    def reload_services(self):
        """Reload all service modules for hot reloading."""
        try:
            from reload_utils import reload_project
            
            # Reload all project modules
            reloaded_count = reload_project()
            
            # Reinitialize services
            self.pdf_parser = pdf_parser.PDFParser()
            self.embedding_service = embedding_service.EmbeddingService()
            self.indexing_service = indexing_service.IndexingService()
            self.qa_service = qa_service.QAService()
            
            print(f"{Fore.GREEN}Successfully reloaded {reloaded_count} modules!{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error reloading services: {str(e)}{Style.RESET_ALL}")

    def run(self):
        """Main loop for the CLI."""
        self.print_welcome()
        
        while True:
            try:
                # Get user input with readline (enables command history)
                command = input(f"\n{Fore.YELLOW}> {Style.RESET_ALL}").strip()
                
                # Add non-empty commands to history
                if command:
                    readline.add_history(command)
                
                # Parse command
                parts = command.split(maxsplit=1)
                cmd = parts[0].lower() if parts else ""
                args = parts[1] if len(parts) > 1 else ""

                # Process commands
                if cmd == "exit":
                    print(f"{Fore.CYAN}Goodbye!{Style.RESET_ALL}")
                    break
                    
                elif cmd == "help":
                    self.print_help()
                    
                elif cmd == "ingest":
                    if not args:
                        print(f"{Fore.RED}Error: Please specify a folder path{Style.RESET_ALL}")
                        continue
                    self.ingest_folder(args)
                    
                elif cmd == "ask":
                    if not args:
                        print(f"{Fore.RED}Error: Please specify a question{Style.RESET_ALL}")
                        continue
                    self.ask_question(args)
                    
                elif cmd == "status":
                    self.show_index_status()    
                    
                elif cmd == "invalidate":
                    if not IS_DEV:
                        print(f"{Fore.RED}Invalidate command is only available in development environment{Style.RESET_ALL}")
                        continue
                    if not args:
                        print(f"{Fore.RED}Error: Please specify a file or folder path{Style.RESET_ALL}")
                        continue
                    self.invalidate_documents(args)
                    
                elif cmd == "reload":
                    if not IS_DEV:
                        print(f"{Fore.RED}Reload command is only available in development environment{Style.RESET_ALL}")
                        continue
                    self.reload_services()
                    
                else:
                    print(f"{Fore.RED}Unknown command. Type 'help' for available commands.{Style.RESET_ALL}")

            except KeyboardInterrupt:
                print("\nUse 'exit' command to quit.")
                continue
                
            except Exception as e:
                print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
                continue

def main():
    """Entry point for the application."""
    try:
        cli = CLI()
        cli.run()
    except Exception as e:
        print(f"{Fore.RED}Fatal error: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)

if __name__ == "__main__":
    main() 