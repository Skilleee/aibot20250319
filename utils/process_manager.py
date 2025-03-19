import logging
import multiprocessing
import time

# Konfigurera loggning
logging.basicConfig(filename="process_manager.log", level=logging.INFO)


def worker(task_name, duration):
    """
    Simulerar en bakgrundsprocess.
    """
    try:
        logging.info(f"✅ Startar process: {task_name}")
        time.sleep(duration)
        logging.info(f"✅ Klar: {task_name}")
    except Exception as e:
        logging.error(f"❌ Fel i process {task_name}: {str(e)}")


def manage_processes(tasks):
    """
    Hanterar flera parallella processer.
    """
    processes = []

    for task_name, duration in tasks:
        process = multiprocessing.Process(target=worker, args=(task_name, duration))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    logging.info("✅ Alla processer klara!")


# Exempelanrop
if __name__ == "__main__":
    task_list = [("Datainsamling", 2), ("Analys", 3), ("Rapportgenerering", 1)]
    manage_processes(task_list)
    print("📊 Alla processer slutförda!")
