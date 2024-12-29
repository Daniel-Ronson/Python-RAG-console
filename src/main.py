import sys
from pathlib import Path
from typing import List, Optional
import click
from colorama import init, Fore, Style
from tqdm import tqdm

from src.core.pdf_parser import PDFParser
from src.core.embedding_service import EmbeddingService
from src.core.indexing_service import IndexingService
from src.core.qa_service import QAService

# Initialize colorama for cross-platform colored output
init()

class CLI:
    def __init__(self):
        self.pdf_parser = PDFParser()
        self.embedding_service = EmbeddingService()
        self.indexing_service = IndexingService()
        self.qa_service = QAService()
        
    def print_welcome(self):
        """Print welcome message and available commands."""
        print(f"{Fore.CYAN}Welcome to Scientific Paper Parser!{Style.RESET_ALL}")
        self.print_help()

    def print_help(self):
        """Print available commands."""
        print("\nAvailable commands:")
        print(f"{Fore.GREEN}ingest <folder>{Style.RESET_ALL} - Parse and index PDFs from a folder")
        print(f"{Fore.GREEN}ask <question>{Style.RESET_ALL} - Ask a question about the indexed papers")
        print(f"{Fore.GREEN}help{Style.RESET_ALL} - Show this help message")
        print(f"{Fore.GREEN}exit{Style.RESET_ALL} - Exit the application")

    def ingest_folder(self, folder_path: str):
        """Ingest all PDFs from a folder."""
        try:
            folder = Path(folder_path)
            if not folder.exists():
                print(f"{Fore.RED}Error: Folder does not exist{Style.RESET_ALL}")
                return

            pdf_files = list(folder.glob("*.pdf"))
            if not pdf_files:
                print(f"{Fore.YELLOW}No PDF files found in the folder{Style.RESET_ALL}")
                return

            print(f"\nProcessing {len(pdf_files)} PDF files...")
            
            success_count = 0
            error_count = 0
            errors = []
            
            for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
                try:
                    # Parse PDF into chunks
                    chunks = self.pdf_parser.parse_pdf(pdf_file)
                    
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
            if success_count == len(pdf_files):
                print(f"{Fore.GREEN}Successfully processed all {success_count} PDF files!{Style.RESET_ALL}")
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

    def run(self):
        """Main loop for the CLI."""
        self.print_welcome()
        
        while True:
            try:
                # Get user input
                command = input(f"\n{Fore.YELLOW}> {Style.RESET_ALL}").strip()
                
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