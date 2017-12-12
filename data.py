# Youri Schuur
# BA Thesis IK
# Data preprocessing

import re

def main():
	
# Make a nested list with all users & tweets in a list
	my_list = [[x for x in line.strip().split('	')] for line in open("test_set.txt","r")]
	
	f = open("final_test_set.txt","w")
	
	for tweets in my_list:
		del(tweets[:2])
		if tweets[1] != "Not Available":
			for words in tweets[1].split():
				# Filter out url's and replace by space
				if "http" in words:
					tweets[1] = tweets[1].replace(words,"")
				
				# Filter out @users and replace by space
				if "@" in words:
					tweets[1] = tweets[1].replace(words,"")
			
				
				# Filter out RT's and replace by space
				if "RT" in words:
					tweets[1] = tweets[1].replace(words,"")
					
				# Filter out other noise
				if "&" in words:
					tweets[1] = tweets[1].replace(words,"")
					
				# Filter out #sarcasm from sarcastic tweets
				"""if "#sarcasm" in words:
					#tweets[1] = tweets[1].replace(words,"")"""
		
			# Remove duplicate spaces
			tweets[1] = re.sub("\s\s+" , " ", tweets[1])
		
			# Remove leading & ending spaces
			tweets[1] = tweets[1].strip()
			
			# convert list to string 
			tweets = '	'.join(tweets)
		
			# Write to file
			f.write(tweets + '\n')
		
	f.close()	
			
main()
