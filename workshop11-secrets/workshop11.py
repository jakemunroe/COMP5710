import hvac
import csv

# Reading CSV file with secrets
with open ('secrets.csv') as csvfile:
    print('Reading CSV file...')
    variables_list = []
    values_list = []
    reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            print(f'\tVariable: {row[0]} \tValue: {row[1]}')
            variables_list.append(row[0])
            values_list.append(row[1])
            line_count += 1
    print(f'Reading done. Processed {line_count} lines.')
    print(f'Variables List: {variables_list} \n Values List: {values_list}')

    
    ###########################################
    #############Password Vault################
    ###########################################

    
    # Getting IP Addr and token from user.
    ip_address = input('Please enter IP Address: ')
    token = input('Please enter token: ')

    hvc_client = hvac.Client(url=ip_address, token=token )

    print('Creating secrets in vault...')
    count = 0
    # Entering Secrets into vault.
    try:
        for variable in variables_list:
            create_response = hvc_client.secrets.kv.v2.create_or_update_secret(path=variables_list[count], secret=dict(password=values_list[count]),)
            print(f'{variables_list[count]} entered.')

            read_response      = hvc_client.secrets.kv.read_secret_version(path=variables_list[count])
            print(f'{variables_list[count]} verified.')

            secret_from_vault  = read_response['data']['data']['password']
            print(f'Value for {variables_list[count]}:{secret_from_vault}\n')

            count += 1
    except Exception as e:
        print(f'An error occured: {e}')
