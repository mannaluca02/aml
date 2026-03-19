

20/04/2016Financial Data description:
http://lisp.vse.cz/pkdd99/Challenge/berka.htm1/6

PKDD'99 Discovery Challenge
Guide to the Financial Data Set


## Domain
Once upon a time, there was a bank offering services to private persons. The services include
managing of accounts, offering loans, etc.
Task description
The bank wants to improve their services. For instance, the bank managers have only vague idea, who
is a good client (whom to offer some additional services) and who is a bad client (whom to watch
carefully to minimize the bank loses). Fortunately, the bank stores data about their clients, the
accounts (transactions within several months), the loans already granted, the credit cards issued The
bank managers hope to improve their understanding of customers and seek specific actions to improve
services. A mere application of a discovery tool will not be convincing for them.

Data description
The data about the clients and their accounts consist of following relations:
relation account (4500 objects in the file ACCOUNT.ASC) ­ each record describes static
characteristics of an account,

20/04/2016Financial Data description:
http://lisp.vse.cz/pkdd99/Challenge/berka.htm2/6
relation client (5369 objects in the file CLIENT.ASC) ­ each record describes characteristics of
a client,
relation disposition (5369 objects in the file DISP.ASC) ­ each record relates together a client
with an account i.e. this relation describes the rights of clients to operate accounts,
relation permanent order (6471 objects in the file ORDER.ASC) ­ each record describes
characteristics of a payment order,
relation transaction (1056320 objects in the file TRANS.ASC) ­ each record describes one
transaction on an account,
relation loan (682 objects in the file LOAN.ASC) ­ each record describes a loan granted for a
given account,
relation credit card (892 objects in the file CARD.ASC) ­ each record describes a credit card
issued to an account,
relation demographic data (77 objects in the file DISTRICT.ASC) ­ each record describes
demographic characteristics of a district.
Each account has both static characteristics (e.g. date of creation, address of the branch) given in
relation "account" and dynamic characteristics (e.g. payments debited or credited, balances) given in
relations "permanent order" and "transaction". Relation "client" describes characteristics of persons
who can manipulate with the accounts. One client can have more accounts, more clients can
manipulate with single account; clients and accounts are related together in relation "disposition".
Relations "loan" and "credit card" describe some services which the bank offers to its clients; more
credit cards can be issued to an account, at most one loan can be granted for an account. Relation
"demographic data" gives some publicly available information about the districts (e.g. the
unemployment rate); additional information about the clients can be deduced from this.

Relation account
itemmeaningremark
account_id
identification of the
account
district_idlocation of the branch
date
date of creating of the
account
in the form YYMMDD
frequency
frequency of issuance of
statements
"POPLATEK MESICNE" stands for monthly
issuance
"POPLATEK TYDNE" stands for weekly
issuance
"POPLATEK PO OBRATU" stands for
issuance after transaction

Relation client
itemmeaningremark
client_idclient identifier

20/04/2016Financial Data description:
http://lisp.vse.cz/pkdd99/Challenge/berka.htm3/6
birth
number
birthday and sex
the number is in the form YYMMDD for men,
the number is in the form YYMM+50DD for
women,
where YYMMDD is the date of birth
district_id
address of the
client

Relation disposition
itemmeaningremark
disp_idrecord identifier
client_ididentification of a client
account_id
identification of an
account
type
type of disposition
## (owner/user)
only owner can issue permanent orders and
ask for a loan

Relation permanent order (debits only)
itemmeaningremark
order_idrecord identifier
account_id
account, the order is issued
for
bank_tobank of the recipienteach bank has unique two­letter code
account_toaccount of the recipient
amountdebited amount
## K_symbol
characterization of the
payment
"POJISTNE" stands for insurrance
payment
"SIPO" stands for household payment
"LEASING" stands for leasing
"UVER" stands for loan payment

## Relation Transaction
itemmeaningremark
trans_idrecord identifier

20/04/2016Financial Data description:
http://lisp.vse.cz/pkdd99/Challenge/berka.htm4/6
account_id
account, the transation
deals with
datedate of transactionin the form YYMMDD
type+/­ transaction
"PRIJEM" stands for credit
"VYDAJ" stands for withdrawal
operationmode of transaction
"VYBER KARTOU" credit card
withdrawal
"VKLAD" credit in cash
"PREVOD Z UCTU" collection from
another bank
"VYBER" withdrawal in cash
"PREVOD NA UCET" remittance to
another bank
amountamount of money
balancebalance after transaction
k_symbol
characterization of the
transaction
"POJISTNE" stands for insurrance payment
"SLUZBY" stands for payment for
statement
"UROK" stands for interest credited
"SANKC. UROK" sanction interest if
negative balance
"SIPO" stands for household
"DUCHOD" stands for old­age pension
"UVER" stands for loan payment
bankbank of the partnereach bank has unique two­letter code
accountaccount of the partner

## Relation Loan
itemmeaningremark
loan_idrecord identifier
account_ididentification of the account
date
date when the loan was
granted
in the form YYMMDD
amountamount of money

20/04/2016Financial Data description:
http://lisp.vse.cz/pkdd99/Challenge/berka.htm5/6
durationduration of the loan
paymentsmonthly payments
statusstatus of paying off the loan
'A' stands for contract finished, no
problems,
'B' stands for contract finished, loan not
payed,
'C' stands for running contract, OK so far,
'D' stands for running contract, client in
debt

Relation Credit card
itemmeaningremark
card_idrecord identifier
disp_iddisposition to an account
typetype of cardpossible values are "junior", "classic", "gold"
issuedissue datein the form YYMMDD

Relation Demographic data
itemmeaningremark
A1 = district_iddistrict code
A2district name
## A3region
A4no. of inhabitants
A5no. of municipalities with inhabitants < 499
A6no. of municipalities with inhabitants 500­1999
A7no. of municipalities with inhabitants 2000­9999
A8no. of municipalities with inhabitants >10000
A9no. of cities
A10ratio of urban inhabitants
A11average salary
A12unemploymant rate '95
A13unemploymant rate '96
A14no. of enterpreneurs per 1000 inhabitants

20/04/2016Financial Data description:
http://lisp.vse.cz/pkdd99/Challenge/berka.htm6/6
A15no. of commited crimes '95
A16no. of commited crimes '96

This database was prepared by Petr Berka and Marta Sochorova.
For possible questions on the data and task description contact Petr Berka. All questions and answers
will be publihked as appendixes to this document.
## Asked Questions