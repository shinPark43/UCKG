import sys
import os
import logging
import time
import network
from process import shared_functions as sf

sys.path.append("./data_collection")
from data_collection import cve_collection as cve, cwe_collection as cwe, d3fend_collection as d3fend, attack_collection as attack, capec_collection as capec

uco_abs_path = os.environ['UCO_ONTO_PATH']
root_folder_abs_path = os.environ['ROOT_FOLDER']

# Configure the logging module
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a logger
logger = logging.getLogger('entry_logger')

# Start the network script to expose scripts to prometheus
network.signal_network_start()

# Waiting for neo4j to startup
logger.info("Waiting 2 minutes for neo4j to startup...")
time.sleep(120)


cwe_data_status = cwe.check_cwe_status()
if cwe_data_status == 3:
    logger.info("The CWE database has not been created yet, starting initialization now...\n")
    cwe.cwe_init()
elif cwe_data_status == 0:
    logger.info("The CWE initialization has not finished yet, continuing now...\n")
    cwe.cwe_init()
    

cve_data_status = cve.check_cve_status()
if cve_data_status == 3:
    logger.info("The CVE database has not been created yet, starting initialization now...\n")
    cve.cve_init()
elif cve_data_status == 0:
    logger.info("The CVE initialization has not finished yet, continuing now...\n")
    cve.cve_init()


d3fend_data_status = sf.check_status("d3fend")
if d3fend_data_status == 3:
    logger.info("The D3FEND database has not been created yet, starting initialization now...\n")
    d3fend.d3fend_init()
elif d3fend_data_status == 0:
    logger.info("The D3FEND initialization has not finished yet, continuing now...\n")
    d3fend.d3fend_init()


attack_data_status = sf.check_status("attack")
if attack_data_status == 3:
    logger.info("The ATT&CK database has not been created yet, starting initialization now...\n")
    attack.attack_init()
elif attack_data_status == 0:
    logger.info("The ATT&CK initialization has not finished yet, continuing now...\n")
    attack.attack_init()


capec_data_status = sf.check_status("capec")
if capec_data_status == 3:
    logger.info("The CAPEC database has not been created yet, starting initialization now...\n")
    capec.capec_init()
elif capec_data_status == 0:
    logger.info("The CAPEC initialization has not finished yet, continuing now...\n")
    capec.capec_init()


logger.info("###############################################")
logger.info("All Data Sources Have Been Initialized!")
logger.info("###############################################")