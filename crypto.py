from cryptography.fernet import Fernet
import argparse
import os

def encrpypt_file(input_file: str, output_file:str , ) -> None:
    # using the generated key
    fernet = Fernet(input('Enter key for encryption: '))
    
    # opening the original file to encrypt
    with open(input_file, 'rb') as file:
        original = file.read()
        
    # encrypting the file
    encrypted = fernet.encrypt(original)
    
    # opening the file in write mode and 
    # writing the encrypted data
    with open(output_file, 'wb') as encrypted_file:
        encrypted_file.write(encrypted)


def decrypt_file(input_file: str, output_file:str , key: str) -> None:
    
    fernet = Fernet(input('Enter key for decryption: '))

    # opening the encrypted file
    with open(input_file, 'rb') as enc_file:
        encrypted = enc_file.read()
    
    # decrypting the file
    decrypted = fernet.decrypt(encrypted)
    
    # opening the file in write mode and
    # writing the decrypted data
    with open(output_file, 'wb') as dec_file:
        dec_file.write(decrypted)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog='Encrypt/Decrypt',
                        description='What the program does',
                        epilog='Text at the bottom of help')
    parser.add_argument('--action', 
                        required=True,
                        type=str,
                        choices=['encrypt', 'decrypt'],
                        dest='action',
                        )
    parser.add_argument('--input', 
                        required=True,
                        type=str,
                        dest='input',
                        )
    parser.add_argument('--output', 
                        required=True,
                        type=str,
                        dest='output',
                        )
    args = parser.parse_args()

    assert os.path.isfile(args.input)
    assert not os.path.isfile(args.output)

    if args.action == 'encrypt':
        encrpypt_file(input_file=args.input, output_file=args.output)
    else:
        assert args.action == 'decrypt'
        decrypt_file(input_file=args.input, output_file=args.output)



