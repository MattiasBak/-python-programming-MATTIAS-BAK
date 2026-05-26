/*
USE master;
If DB_ID('BokhandelDB') is not null
Begin
	Alter Database BokhandelDB set single_user With rollback immediate;
	Drop Database BokhandelDB;
End
Go
*/
Create database BokhandelDB;

GO
Use BokhandelDB;

GO

create table Författare(FörfattareID int identity(1,1) Primary Key,
			Förnamn nvarchar(50) not null,
			Efternamn nvarchar(50) not null,
			Födelsedatum Date
			);
GO

Create table Förlag (FörlagID int identity (1,1) Primary key,
			Förlagname nvarchar(50) not null
			);

GO


Create Table Bok (ISBN char (13) primary key Check (len(ISBN)=13),
			Title nvarchar (100) not null,
			Pris Decimal (10,2) Check (Pris>=0),
			Språk nvarchar(50),
			Utgivningsdatum INT,
			FörlagID INT,
			Foreign key (FörlagID) references Förlag(FörlagID)
			);

GO


Create Table Bokförfattare (ISBN char(13),
			FörfattareID int,
			Primary key (ISBN, FörfattareID),
			Foreign key (ISBN) references Bok(ISBN),
			Foreign key (FörfattareID) References Författare(FörfattareID)
			);

GO

Create Table Butik (ButikID int Identity (1,1) Primary key,
			ButikName nvarchar(100) not null,
			ButikAdress nvarchar(100) not null
			);

GO

Create Table Lagersaldo (ButikID int,
			ISBN char (13),
			Antal int not null Check (Antal>=0),
			Primary key (ButikID, ISBN),
			Foreign key (ButikID) References Butik (ButikID),
			Foreign key (ISBN) references Bok (ISBN)
			);

GO

Create Table Kunder (KunderID int identity(1,1) Primary Key,
			Kundname nvarchar(100),
			Kundtele nvarchar(15)
			);

GO

Create Table Ordrar (Ordernr int identity (1,1) Primary key,
			KundID int,
			ButikID int,
			Orderdatum Date NOT NULL,
			Foreign key (KundID) references Kunder(KunderID),
			Foreign key (ButikID) references Butik(ButikID)
			);

GO

create View TitlarPerFörfattare as
	select Författare.Förnamn+' '+Författare.Efternamn as [Namn],
	Count (Distinct(Bok.ISBN)) as [Titlar],
	Datediff(Year,Författare.Födelsedatum, Getdate()) as [Ålder],
	sum (Isnull (Lagersaldo.antal, 0)) As [Totalt antal i lager],
	Sum(Isnull(Lagersaldo.Antal,0) * (isnull (Bok.pris,0))) as [Lagervärde]
from Författare
left Join BokFörfattare 
	on Författare.FörfattareID = Bokförfattare.FörfattareID
left Join Bok 
	on Bokförfattare.ISBN = Bok.ISBN
Left join Lagersaldo on Bok.ISBN = Lagersaldo.ISBN
Group By 
	Författare.FörfattareID,
	Författare.Förnamn, 
	Författare.Efternamn, 
	Författare.Födelsedatum;

GO

Create table OrderRader (Ordernr int,
			ISBN Char(13),
			Antal int not null Check (Antal>=0),
			Pris Decimal (10,2) Check (Pris>=0),
			Primary key (Ordernr, ISBN),
			Foreign key (Ordernr) references Ordrar (Ordernr),
			Foreign key (ISBN) references Bok(ISBN));

GO

Create View OrderPerButik as
		Select b.ButikID, b.ButikName as [Butiknamn],
		Count(Distinct o.Ordernr) as [AntalOrdrar],
		sum(Isnull(r.Antal, 0))	as [Totalt Sålda Böcker],
		Sum(Isnull(r.Antal,0)* Isnull(r.Pris,0))as [Total Försäljninng (Kronor)]
From Butik b
Left Join Ordrar o On b.ButikID= o.ButikID
Left Join OrderRader r On o.Ordernr = r.Ordernr
Group By b.ButikID,
		b.ButikName;



GO

Create procedure FlyttaBok
			@FromButikID int,
			@ToButikID int,
			@ISBN Char (13),
			@Antal int=1
As
Begin
	set nocount on;
	Begin try
	Begin transaction;
		if NOT EXISTS (
		select 1 from Lagersaldo
		where ButikID = @FromButikID
		and ISBN = @ISBN
		and Antal >= @Antal
		)
Begin
		Throw 50000,'Inte tillräckigt i lager',1;
End
	
	update Lagersaldo
	set Antal=Antal-@Antal
	where ButikID = @FromButikID and ISBN = @ISBN
	if Exists (select 1 from Lagersaldo
				where ButikID = @ToButikID
				and ISBN = @ISBN)
Begin
	update Lagersaldo
	set Antal=Antal+@Antal
	where ButikID = @ToButikID and ISBN = @ISBN
End
Else
Begin
	insert into Lagersaldo(ButikID, ISBN, Antal)
	Values (@ToButikID,@ISBN,@Antal);
End

	Commit Transaction;
End Try
Begin Catch
	if @@TRANCOUNT>0
		Begin
		Rollback Transaction;
		End;
		Throw;

End Catch
END;
