import subprocess as sb

# required_packages = ['mutagen', 'gTTS']

# installed_packages = [package.strip() 
#                       for package in sb.check_output(['pip', 'freeze'], universal_newlines=True)]

# missing_packages = [package for package in required_packages
#                      if package not in installed_packages]

# if missing_packages:
#     print(f"Missing packages: {missing_packages}")
#     sb.check_call(['pip', 'install'] + missing_packages)
# else:
#     print("All required packages are installed.")

def need_packages(rq_pckg:list[str,]):
    inst_pckgs = [pckg.strip()
                 for pckg in sb.check_output(['pip', 'freeze'],
                                              universal_newlines=True)]
    miss_pckgs = [pckg for pckg in rq_pckg 
                  if pckg not in inst_pckgs]
    if miss_pckgs :
        print(f"Missing packages: {miss_pckgs}")
        sb.check_call(['pip', 'install'] + miss_pckgs)
    else:
        return None