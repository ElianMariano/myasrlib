from configparser import ConfigParser

def read_config(filename='config.ini', section='config'):
    parser = ConfigParser()
    parser.read(filename)

    entries = {}
    if parser.has_section(section):
        params = parser.items(section)
        
        for param in params:
            entries[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
    
    return entries