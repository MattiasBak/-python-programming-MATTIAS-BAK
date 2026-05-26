-- Moon Missions

drop table if exists SuccessfulMissions
Select [Spacecraft], 
	[Launch date], 
	[Carrier rocket], 
	[Operator], 
	[Mission type]
into SuccessfulMissions 
from MoonMissions
Where Outcome like 'Success%'

GO

update [SuccessfulMissions]
SET Operator = TRIM ([Operator])

GO

update [SuccessfulMissions]
SET Spacecraft = LEFT (Spacecraft, CHARINDEX ('(', Spacecraft+ '(' )-1)
	where Spacecraft like '%(%';

GO

select [Operator], [Mission type], 
	count (*) as [Mission Count]
	from [SuccessfulMissions]
	Group by [Operator], [Mission type]
Having count (*)>1
	order by [Mission count] desc, [Operator], [Mission type] 


GO

--Users

select * from Users


select *,
	[FirstName]+' '+[LastName] AS [Name],
	CASE
		When cast (substring(ID, len(ID)-1,1) as int)%2 =0
		then 'Female'
		else 'Male'
	END AS Gender
into NewUsers
From Users;

GO

Alter Table NewUsers
Alter Column Username nvarchar(50);

GO

Select UserName, 
	count (*) as Dupl
from NewUsers
Group by UserName	
	Having Count (*) >1
	Order by [Username] desc


GO

WITH Dupl AS (
    SELECT
        Username,
        ROW_NUMBER() OVER (
            PARTITION BY Username
            ORDER BY Username
        ) AS rn
    FROM NewUsers
)
UPDATE Dupl
SET Username = Username + CAST(rn AS varchar(10))
WHERE rn > 1


GO

Delete from NewUsers
	Where Gender = 'Female' 
	and 
	(CASE 
		When len(ID)>=12 Then cast(left(ID,4)as int)
		When cast(left(ID,2)as int)>Year(Getdate())%100
			Then 1900+cast(left(ID,2)as int)
			else 2000+cast(left(id,2)as int)
		end	
			)<1970;
		
	

GO

insert into NewUsers (ID, UserName, [Password], FirstName, LastName, Email, Phone, [Name], Gender)
Values ('930813-4866', 'eriand', 'hymbrgd1#53', 'Erik', 'Andersson',	'E.andersson@hotmail.se',
		'070-3666825', 'Erik Andersson', 'Male')


GO

select Gender, 
	AVG(
		YEAR (GETDATE()) -
		CASE
			When len(ID)>=12 Then cast(left(ID,4)as int)
			when cast(left(ID,2)as int)>Year(Getdate())%100
			Then 1900+cast(left(ID,2)as int)
			else 2000+cast(left(id,2)as int)
		end
		) AS [average age]

from NewUsers
Group By Gender;


GO

--Company

Select company.products.id, 
		company.products.ProductName as [Product],
		company.suppliers.CompanyName as [Company],
	    company.categories.CategoryName as[Category]
		from company.products
		join company.suppliers
		on company.products.SupplierId = company.suppliers.Id
	    join company.categories
		on company.products.CategoryId = company.categories.Id;

GO

select company.regions.Id, 
	company.regions.RegionDescription,
	Count (distinct company.employee_territory.EmployeeId) as EmployeeCount
	from Company.regions
	join company.territories
	on company.regions.id = company.territories.RegionId
	join company.employee_territory
	on company.territories.Id= company.employee_territory.TerritoryId
	GROUP BY
		company.regions.id,
		company.regions.regiondescription;

GO

select 
	employee.Id,
	employee.FirstName,
	employee.LastName, 
	employee.Title,
	employee.TitleOfCourtesy+' '+employee.FirstName+' '+employee.LastName as [Name],
	Case
		when employee.reportsto is null then 'Nobody'
		else Manager.title+' '+Manager.FirstName+' '+Manager.LastName
	end as [Reports to]
	
	from company.employees as employee
	Left Join company.employees as Manager
		on employee.ReportsTo = Manager.Id

	