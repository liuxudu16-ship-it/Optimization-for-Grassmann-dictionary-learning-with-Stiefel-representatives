arch=computer;
mac=strcmp(arch,'MACI64') || strcmp(arch,'MACI') || strcmp(arch,'MAC') || strcmp(arch,'MACA64');
windows=strcmp(arch,'PCWIN64') || strcmp(arch,'PCWIN');
linux=strcmp(arch,'GLNXA64') || strcmp(arch,'GLNX86');
sixtyfourbits=strcmp(arch,'MACI64') || strcmp(arch,'MACA64') || strcmp(arch,'GLNXA64') || strcmp(arch,'PCWIN64');
