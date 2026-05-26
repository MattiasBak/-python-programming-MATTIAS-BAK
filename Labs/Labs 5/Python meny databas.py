from sqlalchemy import create_engine, text
SERVER = '(localdb)\MSSQLLocalDB'
DATABASE = 'BokhandelDB'

connection_url= f"mssql+pyodbc://@{SERVER}/{DATABASE}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes&MultipleActiveResultSets=True"

engine = create_engine(connection_url)
    
def visa_orderstatestik():
    print ("\n               --- ORDERSTATESTIK PER BUTIK ---")
    try:
        with engine.connect() as connection:
            query = text("SELECT * FROM OrderPerButik")
            result = connection.execute(query)
            rows = result.fetchall()
            if not rows:
                print ("[INFO] inga rader hittades")
                return
            print (f"{'ID':<4}{'Butiknamn':<20}{'Ordrar':<8}{'Sålda Böcker':<15}{'Försäljning':<15}")
            print("-"*65)
            for row in rows:
                print (f"{row[0]:<4}{row[1]:<20}{row[2]:<8}{row[3]:<15}{float(row[4]):<15.2f}")
    except Exception as e:
        print (f"Ett fel uppstod vid hämtningen av :{e}")

def flytta_bok_mellan_lager():
    print(f"\n--- Flytta Böcker Mellan Lager ---")
    try:
        from_butik = int(input("Från ButikID\n>"))
        to_butik = int(input("Till ButikID\n>"))
        isbn = input("Bokens ISBN (13 siffror)\n>").strip()
        antal = int(input("Antal böcker som skall flyttas\n>"))
    except ValueError:
        print ("[FEL] Input måste vara siffror där det efterfrågas.")
        return
    try:
        with engine.connect() as conn:
            titel_query = text ('Select Title From Bok Where ISBN = :isbn')
            title_result = conn.execute(titel_query,{'isbn': isbn})
            row = title_result.fetchone()

            if row:
                title = row[0]
            else:
                title= "Okänd titel (felaktig ISBN)"
            
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:    
            sql_command = text("EXEC FlyttaBok :from_id, :to_id, :isbn, :antal")
            connection.execute(sql_command, 
                               {'from_id': from_butik, 
                                'to_id': to_butik,
                                'isbn': isbn, 
                                'antal': antal})
            print(f"\n[SUCCESS]\nFörflyttningen av {antal}st exemplar utav {title} lyckades")

    except Exception as e:
        print("\n[DATABAS-MEDDELANDE]")
        error_msg = str(e).split(']')[-1] if ']' in str(e) else str(e)
        print (error_msg)

def huvudmeny():
    while True:
        print("\n---------------------------------")
        print("    BOKHANDEL MANAGEMENT SYSTEM  ")
        print("----------------------------------")
        print("1. Visa försäljningsstatestik (Vy)")
        print("2. Flytta bok mellan butiker")
        print("0. Avsluta programmet")
        val = input("\n\nVälj ett av alternativen\n>")
        if val == "1":
            visa_orderstatestik()
        elif val =="2":
            flytta_bok_mellan_lager()
        elif val =="0":
            break
        else:
            print("\n Ogiltingt val")

if __name__== "__main__":
    huvudmeny()