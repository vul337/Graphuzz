from cy.nn_server import NNModel
from cy.nn_server import NNServer
from cy.exportTrainingSet import CDFG

#import pyroscope
#
#pyroscope.configure(
#  application_name = "gfuzz.fuzzer.nn_server", # replace this with some name for your application
#  server_address   = "http://172.17.0.1:4040", # replace this with the address of your pyroscope server
#)

if __name__ == "__main__":
    nn_model = NNModel()
    nn_server = NNServer(nn_model)
    nn_server.main_loop()

